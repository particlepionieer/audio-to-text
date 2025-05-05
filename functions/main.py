import logging
import json
from firebase_functions import storage_fn
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize clients
speech_client = None
storage_client = None

# Common sample rates for different types of audio formats
SAMPLE_RATES = {
    'wav': [16000, 44100, 48000],  # WAV files can have multiple common sample rates
    'flac': [16000, 44100, 48000],  # FLAC files can have multiple common sample rates
    'mp3': [44100, 48000],  # MP3 files commonly use 44100 or 48000 Hz
    'ogg': [16000, 44100, 48000],  # OGG files commonly use 16000, 44100, or 48000 Hz
    'webm': [16000, 44100, 48000],  # WebM files commonly use 16000, 44100, or 48000 Hz
    'm4a': [44100, 48000],  # M4A files commonly use 44100 or 48000 Hz
}


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


def get_sample_rate(file_extension):
    """Get the sample rate(s) for the given audio file extension."""
    return SAMPLE_RATES.get(file_extension.lower(), [16000])  # Default to 16000 if unknown extension


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

    # Extract the file extension and get available sample rates
    file_extension = file_name.split('.')[-1]
    available_sample_rates = get_sample_rate(file_extension)

    if not available_sample_rates:
        logging.warning(f"No valid sample rate found for {file_extension} file type.")
        return

    # Construct the Cloud Storage URI for the audio file
    gcs_uri = f"gs://{bucket_name}/{file_name}"

    # Configure the speech recognition request for different formats and sample rates
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Default to LINEAR16 for most formats
        sample_rate_hertz=available_sample_rates[0],  # Default to the first available sample rate
        language_code='en-US',  # Set to 'en-US' as the default, will change based on multi-language setting
        alternative_language_codes=[
            'fr-FR',  # French
            'pt-BR',  # Portuguese (Brazil)
            'pt-PT',  # Portuguese (Portugal)
            'nl-NL',  # Dutch
            'de-DE',  # German
            'es-ES',  # Spanish
            'it-IT',  # Italian
            # Add other languages you want to support
        ],  # List of alternative languages to allow language auto-detection
        enable_word_time_offsets=True,  # To get word timestamps
        enable_speaker_diarization=True,  # Enable speaker diarization
        diarization_speaker_count=2,  # Set number of expected speakers (adjust as needed)
    )

    audio = speech.RecognitionAudio(uri=gcs_uri)

    # Use long-running recognize for larger audio files
    logging.info(f"Sending {gcs_uri} to Speech-to-Text for transcription...")
    operation = speech_client_instance.long_running_recognize(config=config, audio=audio)

    logging.info("Waiting for transcription operation to complete...")
    response = operation.result(timeout=300)  # Adjust timeout as needed

    # Process the transcription results
    transcript_parts = []
    speaker_turns = []  # This will store the start and end times of each speaker's turn

    current_speaker = None
    turn_start_time = None

    for result in response.results:
        if result.alternatives:
            transcript_parts.append(result.alternatives[0].transcript)

            # Process speaker diarization information
            for word_info in result.alternatives[0].words:
                speaker_tag = word_info.speaker_tag

                # Start a new turn for the speaker if it's different from the previous one
                if speaker_tag != current_speaker:
                    if current_speaker is not None:  # If there's a previous speaker, save their turn
                        speaker_turns.append({
                            'speaker': current_speaker,
                            'start_time': turn_start_time,
                            'end_time': word_info.start_time.seconds + word_info.start_time.nanos * 1e-9,
                        })

                    # Start a new speaker turn
                    current_speaker = speaker_tag
                    turn_start_time = word_info.start_time.seconds + word_info.start_time.nanos * 1e-9

            # Add the last speaker turn at the end of the file
            if current_speaker is not None:
                speaker_turns.append({
                    'speaker': current_speaker,
                    'start_time': turn_start_time,
                    'end_time': word_info.end_time.seconds + word_info.end_time.nanos * 1e-9,
                })

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

    # Save the speaker turn timestamps to a separate file
    timestamp_file_name = file_name + "_speaker_turns.json"
    timestamp_blob = storage_client_instance.bucket(bucket_name).blob(timestamp_file_name)

    try:
        timestamp_blob.upload_from_string(
            json.dumps({'speaker_turns': speaker_turns}),
            content_type="application/json"
        )
        logging.info(f"Speaker turns saved to gs://{bucket_name}/{timestamp_file_name}")
    except Exception as e:
        logging.error(f"Failed to save speaker turns: {e}")

    logging.info(f"Finished processing {file_name}.")