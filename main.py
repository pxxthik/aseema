from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import base64
import logging
from datetime import datetime
import json

# Import your existing modules
from dotenv import load_dotenv
load_dotenv()

# Audio processing
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# AI services
from groq import Groq
from gtts import gTTS
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
    
    def text_to_speech_gtts(self, text, output_path):
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            return output_path
        except Exception as e:
            logging.error(f"gTTS error: {e}")
            return None
    
    def text_to_speech_elevenlabs(self, text, output_path):
        """Convert text to speech using ElevenLabs (if API key available)"""
        try:
            if not hasattr(self, 'elevenlabs_client'):
                return self.text_to_speech_gtts(text, output_path)
                
            audio = self.elevenlabs_client.generate(
                text=text,
                voice="Rachel",  # A clear, friendly voice for educational content
                output_format="mp3_22050_32",
                model="eleven_turbo_v2"
            )
            elevenlabs.save(audio, output_path)
            return output_path
        except Exception as e:
            logging.error(f"ElevenLabs error: {e}")
            # Fallback to gTTS
            return self.text_to_speech_gtts(text, output_path)

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
        
        # Use ElevenLabs if available, otherwise gTTS
        audio_output = ai_assistant.text_to_speech_elevenlabs(ai_response, response_audio_path)
        
        # Clean up uploaded files
        os.remove(audio_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        # Log interaction
        logging.info(f"Voice interaction - Student said: {transcribed_text[:100]}...")
        
        return jsonify({
            'transcribed_text': transcribed_text,
            'ai_response': ai_response,
            'audio_response_url': f'/audio/{response_audio_filename}',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Voice chat error: {e}")
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

if __name__ == '__main__':
    print("Starting Aseema Educational AI Assistant...")
    print("Service available 24/7 for student support")
    app.run(debug=True, host='0.0.0.0', port=5000)
