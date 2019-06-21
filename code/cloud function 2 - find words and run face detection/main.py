import json

import cv2
from google.cloud import storage
from google.cloud import vision

from google.protobuf.json_format import MessageToJson

vison_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()


video_bucket = storage_client.get_bucket("swear-jar")
json_bucket = storage_client.get_bucket("swear-jar-output")
json_output_bucket = storage_client.get_bucket("tech-jar-output")

debug_bucket = storage_client.get_bucket("swear-jar-debug")

debug_i = 0

def get_faces(event, context):
    
    # get file info
    json_file = event["name"]
    video_file = event["name"].replace(".json","")

    print("using json file: " + json_file)
    print("using video file: " + video_file)

    json_blob = json_bucket.blob(json_file)
    video_blob = video_bucket.blob(video_file)

    # get the transcription json
    transcript_json = get_json(json_blob)

    print(transcript_json)

    # find all of the trigger words in teh transcript
    trigger_words = get_trigger_word_times(transcript_json)

    print(trigger_words)
    
    # download the video so that we can run it through open-cv
    file_name = None
    with open('/tmp/temp_vid', 'wb') as file_obj:
        file_name = file_obj.name
        print("read video using cv2")
        video_blob.download_to_file(file_obj)
        
    # run open-cv on video 
    video = cv2.VideoCapture(file_name)
        
    seconds, frame_count, fps = get_duration_frames(video)
    print('vid stats')
    print(seconds, frame_count, fps)

    current_frame = 0

    # for each trigger word found, get the frame in the video
    for word in trigger_words:
        start = word['start']
        frames_of_word = int(fps * start)

        frames_till_word = frames_of_word - current_frame

        for _ in range(frames_till_word-1):
            current_frame += 1
            video.read()

        # temporarily save frame so that it can be sent to vision api for face detection  
        success, image = video.read() 
        cv2.imwrite("/tmp/temp_frame.jpg", image)

        # use Vision API do fetch face coordinates
        face = get_face_data("/tmp/temp_frame.jpg")

        word['face_data'] = face

    # save all the face and word info to output bucket
    save_json(trigger_words, json_file)

    print(f"Finished processing file.")

def get_face_data(frame):

    with open(frame, 'rb') as image_file:
        content = image_file.read()

    save_debug_image(frame)
    image = vision.types.Image(content=content)
    response = vison_client.face_detection(image=image)
    faces = json.loads(MessageToJson(response))
    return faces

def get_trigger_word_times(transcript):
    '''
    Given the transcript JSON, return a list of the trigger-words 
    found along with the time information of when they are uttered.
    '''
    trigger_words = []
    sentences = transcript['annotation_results'][0]['speech_transcriptions']
    for sentence in sentences:
        words = sentence['alternatives'][0]['words']
        for word in words:
            if is_trigger_word(word['word']):
                trigger_words.append({"word": word['word'], **get_word_seconds(word)}) 
    return trigger_words

TRIGGER_LIST = {'apis', 'api', 'google', 'machine', 'learning','ai','gcp','android', 'firebase', 'sdk', 'automl'
                , 'sdks', 'cloud', 'database', 'platform', 'realtime','sre', 'seo', 'tensorflow', 'bigquery'} 

def is_trigger_word(word):
    
    # is accronym
    if (len(word) > 1) and (word.isupper()):
        return True
    
    return (word.lower() in TRIGGER_LIST)
    

def get_word_seconds(word):
    start = 0
    start += word['start_time'].get('seconds', 0)
    start += word['start_time'].get('nanos', 0) / 1e+9
    
    end = 0
    end += word['end_time'].get('seconds', 0)
    end += word['end_time'].get('nanos', 0) / 1e+9

    return {"start": start,"end": end}

def get_duration_frames(video):
    print('vid opend: ' + str(video.isOpened()) )
    fps = video.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('vid deets')
    print(fps, frame_count)
    duration = frame_count/fps

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))

    return duration, frame_count, fps

def get_json(json_blob):
    return json.loads(json_blob.download_as_string())

def save_debug_image(image):
    global debug_i
    debug_i += 1
    storage_client.get_bucket("swear-jar-output")
    blob = storage.Blob(str(debug_i), debug_bucket)
    blob.upload_from_filename(image)

def save_json(data, name):
    blob = storage.Blob(name, json_output_bucket)
    blob.upload_from_string(json.dumps(data), content_type="application/json")