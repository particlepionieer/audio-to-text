import logging
from firebase_functions import storage_fn
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize clients
speech_client = None
storage_client = None

def get_speech_client():
    """Gets or creates a SpeechClient, initializing only once."""
    global speech_client
    if speech_client is None:
        logging.info("Initializing SpeechClient...")
        speech_client = speech.SpeechClient()
    return speech_client

def get_storage_client():
    """Gets or creates a StorageClient, initializing only once."""
    global storage_client
    if storage_client is None:
        logging.info("Initializing StorageClient...")
        storage_client = storage.Client()
    return storage_client

# Cloud Function triggered by new object finalized in storage
@storage_fn.on_object_finalized(region="us-east1")
def transcribe_audio(event: storage_fn.StorageObjectData):
    """Triggered when an audio file is uploaded to Cloud Storage."""
    speech_client_instance = get_speech_client()
    storage_client_instance = get_storage_client()

    # Get file details from the event (using dot notation)
    bucket_name = event.data.bucket
    file_name = event.data.name
    content_type = event.data.content_type  # Fixed here

    if not file_name:
        logging.warning("No file name provided in the event.")
        return

    logging.info(f"Processing file: {file_name} in bucket: {bucket_name}")

    # Skip non-audio files
    if content_type and not content_type.startswith('audio/'):
        logging.info(f"Skipping non-audio file: {file_name} ({content_type})")
        return

    # Construct the Cloud Storage URI for the audio file
    gcs_uri = f"gs://{bucket_name}/{file_name}"

    # Configure the speech recognition request for .wav (LINEAR16)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # LINEAR16 for .wav files
        sample_rate_hertz=44100,  # Adjust based on your .wav file's sample rate, often 16000
        language_code="en-US",  # Change as needed
    )
    audio = speech.RecognitionAudio(uri=gcs_uri)

    # Use long-running recognize for larger audio files
    logging.info(f"Sending {gcs_uri} to Speech-to-Text for transcription...")
    operation = speech_client_instance.long_running_recognize(config=config, audio=audio)

    logging.info("Waiting for transcription operation to complete...")
    response = operation.result(timeout=300)  # Adjust timeout as needed

    # Process the transcription results
    transcript_parts = []
    for result in response.results:
        if result.alternatives:
            transcript_parts.append(result.alternatives[0].transcript)

    full_transcript = "\n".join(transcript_parts)

    logging.info(f"Transcription complete. Full transcript:\n{full_transcript}")

    # Determine the output file path for the transcript
    transcript_file_name = file_name + ".txt"
    transcript_blob = storage_client_instance.bucket(bucket_name).blob(transcript_file_name)

    # Save the transcript back to Cloud Storage
    try:
        transcript_blob.upload_from_string(full_transcript, content_type="text/plain")
        logging.info(f"Transcript saved to gs://{bucket_name}/{transcript_file_name}")
    except Exception as e:
        logging.error(f"Failed to save transcript: {e}")

    logging.info(f"Finished processing {file_name}.")
