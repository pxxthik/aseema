from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import base64
import logging
from datetime import datetime
import json
import asyncio

# Import your existing modules
from dotenv import load_dotenv
load_dotenv()

# Audio processing
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# AI services
from groq import Groq
import edge_tts  # Replacing gTTS with edge-tts
import elevenlabs
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_FOLDER'] = 'audio_responses'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")

# Educational AI Assistant System Prompt
EDUCATIONAL_SYSTEM_PROMPT = """
You are an AI Educational Assistant for Aseema NGO, helping students with their learning journey 24/7. 

Your role:
- Be encouraging, patient, and supportive
- Provide clear, age-appropriate explanations
- Help with homework, concepts, and study guidance
- Encourage curiosity and critical thinking
- Be culturally sensitive and inclusive
- If you see study materials or homework in images, help explain concepts
- Always maintain a positive, educational tone

Remember: You're here to guide learning, not just give answers. Help students understand the 'why' behind concepts.
"""

class EducationalAIAssistant:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        if ELEVENLABS_API_KEY:
            self.elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
    def transcribe_audio(self, audio_filepath):
        """Convert speech to text using Groq Whisper"""
        try:
            with open(audio_filepath, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language="en"
                )
            return transcription.text
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return None
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Image encoding error: {e}")
            return None
    
    def analyze_with_vision(self, query, image_path=None):
        """Analyze text query with optional image using Groq Vision model"""
        try:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": EDUCATIONAL_SYSTEM_PROMPT + "\n\nStudent Query: " + query}]
            }]
            
            if image_path:
                encoded_image = self.encode_image(image_path)
                if encoded_image:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        }
                    })
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Vision analysis error: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again."
    
    async def text_to_speech_edge_tts(self, text, output_path, voice="en-US-JennyNeural"):
        """Convert text to speech using Edge TTS"""
        try:
            # Validate inputs
            if not text or not text.strip():
                logging.error("Empty text provided for TTS")
                return None
                
            if len(text) > 3000:  # Edge TTS has limits
                logging.warning(f"Text too long ({len(text)} chars), truncating to 3000")
                text = text[:3000] + "..."
            
            # Log the attempt
            logging.info(f"Creating Edge TTS communication object")
            logging.info(f"Text length: {len(text)} characters")
            logging.info(f"Voice: {voice}")
            
            # Create Edge TTS communication object
            communicate = edge_tts.Communicate(text, voice)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the audio file
            logging.info(f"Saving audio to: {output_path}")
            await communicate.save(output_path)
            
            # Verify the file was created and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logging.info(f"Audio file created successfully. Size: {file_size} bytes")
                if file_size == 0:
                    logging.error("Audio file is empty")
                    return None
                return output_path
            else:
                logging.error("Audio file was not created")
                return None
                
        except Exception as e:
            logging.error(f"Edge TTS error: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def text_to_speech_edge_tts_sync(self, text, output_path, voice="en-US-JennyNeural"):
        """Synchronous wrapper for Edge TTS with proper error handling"""
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Log the attempt
            logging.info(f"Attempting to generate audio for text: {text[:50]}...")
            logging.info(f"Output path: {output_path}")
            logging.info(f"Voice: {voice}")
            
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to run in thread
                logging.info("Running in async context, using thread executor")
                
                def run_async_tts():
                    # Create new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.text_to_speech_edge_tts(text, output_path, voice))
                    finally:
                        new_loop.close()
                
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_tts)
                    result = future.result(timeout=30)  # 30 second timeout
                    
            except RuntimeError:
                # No event loop is running, we can use asyncio.run directly
                logging.info("No running event loop, using asyncio.run")
                result = asyncio.run(self.text_to_speech_edge_tts(text, output_path, voice))
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logging.info(f"Audio file successfully created: {output_path} ({os.path.getsize(output_path)} bytes)")
                return output_path
            else:
                logging.error(f"Audio file was not created or is empty: {output_path}")
                return None
                
        except Exception as e:
            logging.error(f"Edge TTS sync wrapper error: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def text_to_speech_simple_sync(self, text, output_path, voice="en-US-JennyNeural"):
        """Simple synchronous TTS using subprocess call to edge-tts"""
        try:
            import subprocess
            import tempfile
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Log the attempt
            logging.info(f"Using simple sync TTS for text: {text[:50]}...")
            logging.info(f"Output path: {output_path}")
            
            # Create a temporary text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                temp_text_file = f.name
            
            try:
                # Use subprocess to call edge-tts
                cmd = [
                    'edge-tts',
                    '--voice', voice,
                    '--file', temp_text_file,
                    '--write-media', output_path
                ]
                
                logging.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logging.info(f"Audio generated successfully via subprocess: {output_path} ({os.path.getsize(output_path)} bytes)")
                        return output_path
                    else:
                        logging.error("Subprocess completed but no audio file created")
                        return None
                else:
                    logging.error(f"edge-tts subprocess failed: {result.stderr}")
                    return None
                    
            finally:
                # Clean up temp file
                if os.path.exists(temp_text_file):
                    os.unlink(temp_text_file)
                    
        except subprocess.TimeoutExpired:
            logging.error("edge-tts subprocess timed out")
            return None
        except FileNotFoundError:
            logging.error("edge-tts command not found. Please install: pip install edge-tts")
            return None
        except Exception as e:
            logging.error(f"Simple sync TTS error: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def text_to_speech_elevenlabs(self, text, output_path):
        """Convert text to speech using ElevenLabs (if API key available)"""
        try:
            # Remove the 4/0 line that was causing issues
            if not hasattr(self, 'elevenlabs_client'):
                logging.info("ElevenLabs client not available, using Edge TTS")
                return self.text_to_speech_simple_sync(text, output_path)  # Use the new simple sync method
                
            logging.info("Using ElevenLabs for TTS")
            audio = self.elevenlabs_client.generate(
                text=text,
                voice="Rachel",  # A clear, friendly voice for educational content
                output_format="mp3_22050_32",
                model="eleven_turbo_v2"
            )
            elevenlabs.save(audio, output_path)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logging.info(f"ElevenLabs audio created: {output_path}")
                return output_path
            else:
                logging.error("ElevenLabs created empty file")
                return None
                
        except Exception as e:
            logging.error(f"ElevenLabs error: {e}")
            # Fallback to simple sync Edge TTS
            logging.info("Falling back to Edge TTS")
            return self.text_to_speech_simple_sync(text, output_path)

# Initialize the AI assistant
ai_assistant = EducationalAIAssistant()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle text-based chat"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get AI response
        ai_response = ai_assistant.analyze_with_vision(user_message)
        
        # Log interaction
        logging.info(f"Student query: {user_message[:100]}...")
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    """Handle voice-based interaction with optional image"""
    try:
        # Handle audio file
        audio_file = request.files.get('audio')
        image_file = request.files.get('image')
        
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        # Save audio file
        audio_filename = secure_filename(f"audio_{datetime.now().timestamp()}.wav")
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)
        
        # Save image file if provided
        image_path = None
        if image_file:
            image_filename = secure_filename(f"image_{datetime.now().timestamp()}.jpg")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            image_file.save(image_path)
        
        # Transcribe audio
        transcribed_text = ai_assistant.transcribe_audio(audio_path)
        if not transcribed_text:
            return jsonify({'error': 'Could not transcribe audio'}), 400
        
        # Get AI response
        ai_response = ai_assistant.analyze_with_vision(transcribed_text, image_path)
        
        # Convert response to speech
        response_audio_filename = f"response_{datetime.now().timestamp()}.mp3"
        response_audio_path = os.path.join(app.config['AUDIO_FOLDER'], response_audio_filename)
        
        # Try different TTS methods with fallbacks
        audio_output = None
        
        # Method 1: Try ElevenLabs (if available)
        if ELEVENLABS_API_KEY:
            logging.info("Attempting ElevenLabs TTS")
            audio_output = ai_assistant.text_to_speech_elevenlabs(ai_response, response_audio_path)
        
        # Method 2: If ElevenLabs failed or not available, try simple sync Edge TTS
        if not audio_output:
            logging.info("Attempting simple sync Edge TTS")
            audio_output = ai_assistant.text_to_speech_simple_sync(ai_response, response_audio_path)
        
        # Method 3: If simple sync failed, try the original async method
        if not audio_output:
            logging.info("Attempting original Edge TTS sync wrapper")
            audio_output = ai_assistant.text_to_speech_edge_tts_sync(ai_response, response_audio_path)
        
        # Clean up uploaded files
        try:
            os.remove(audio_path)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        except Exception as cleanup_error:
            logging.warning(f"Cleanup error: {cleanup_error}")
        
        # Log interaction
        logging.info(f"Voice interaction - Student said: {transcribed_text[:100]}...")
        
        # Check if audio was generated successfully
        if audio_output and os.path.exists(audio_output):
            return jsonify({
                'transcribed_text': transcribed_text,
                'ai_response': ai_response,
                'audio_response_url': f'/audio/{response_audio_filename}',
                'timestamp': datetime.now().isoformat(),
                'tts_method': 'success'
            })
        else:
            # Return response without audio if TTS failed
            logging.error("All TTS methods failed, returning text-only response")
            return jsonify({
                'transcribed_text': transcribed_text,
                'ai_response': ai_response,
                'audio_response_url': None,
                'timestamp': datetime.now().isoformat(),
                'tts_method': 'failed',
                'error': 'Audio generation failed, but text response available'
            })
    
    except Exception as e:
        logging.error(f"Voice chat error: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    try:
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype='audio/mpeg')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        logging.error(f"Audio serving error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload with text query"""
    try:
        image_file = request.files.get('image')
        query = request.form.get('query', 'Please help me understand this image.')
        
        if not image_file:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Save image file
        image_filename = secure_filename(f"homework_{datetime.now().timestamp()}.jpg")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
        
        # Analyze image with query
        ai_response = ai_assistant.analyze_with_vision(query, image_path)
        
        # Clean up
        os.remove(image_path)
        
        # Log interaction
        logging.info(f"Image analysis - Query: {query[:100]}...")
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Image upload error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Aseema Educational AI Assistant',
        'timestamp': datetime.now().isoformat()
    })

# Route to get available Edge TTS voices (optional - for voice selection feature)
@app.route('/voices')
def get_voices():
    """Get available Edge TTS voices"""
    try:
        # Get list of available voices
        voices = asyncio.run(edge_tts.list_voices())
        
        # Filter for English voices and format for frontend
        english_voices = []
        for voice in voices:
            if voice['Locale'].startswith('en'):
                english_voices.append({
                    'name': voice['Name'],
                    'display_name': voice['FriendlyName'],
                    'gender': voice['Gender'],
                    'locale': voice['Locale']
                })
        
        return jsonify({'voices': english_voices})
    
    except Exception as e:
        logging.error(f"Voice listing error: {e}")
        return jsonify({'error': 'Could not fetch voices'}), 500

if __name__ == '__main__':
    print("Starting Aseema Educational AI Assistant...")
    print("Service available 24/7 for student support")
    print("Using Edge TTS for high-quality voice synthesis")
    app.run(debug=True, host='0.0.0.0')
