from google.cloud import videointelligence

video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.SPEECH_TRANSCRIPTION]

config = videointelligence.types.SpeechTranscriptionConfig(
    language_code='en-US',
    enable_automatic_punctuation=True,
	filter_profanity=True,
    speech_contexts=[videointelligence.types.SpeechContext(phrases=['AutoML'])]
)

video_context = videointelligence.types.VideoContext(
    speech_transcription_config=config)

def find_swears(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    
    input_file = f"gs://{event['bucket']}/{file['name']}"
    output_file = f"gs://swear-jar-output/{file['name']}.json"
    
    print(f"V4 Processing file: {input_file}.")
    print(event)
    
    print(f"Start processing")
    
    operation = video_client.annotate_video(
    input_file, 
    features=features,
    video_context=video_context,
    output_uri=output_file)
    
    def finished(_):
        print("finnished annotation")
    
    operation.add_done_callback(finished)