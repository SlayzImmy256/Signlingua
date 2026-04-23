"""Multilingual translation module"""

from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
from typing import Optional, Dict, List
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Language codes mapping
LANGUAGE_CODES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese (Simplified)': 'zh-CN',
    'Chinese (Traditional)': 'zh-TW',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Turkish': 'tr',
    'Dutch': 'nl',
    'Polish': 'pl',
    'Swedish': 'sv',
    'Norwegian': 'no',
    'Danish': 'da',
    'Finnish': 'fi',
    'Greek': 'el',
    'Hebrew': 'he',
    'Thai': 'th',
    'Vietnamese': 'vi',
    'Indonesian': 'id',
    'Malay': 'ms',
    'Filipino': 'fil',
    'Swahili': 'sw',
    'Urdu': 'ur',
    'Persian': 'fa',
    'Ukrainian': 'uk',
    'Czech': 'cs',
    'Romanian': 'ro',
    'Hungarian': 'hu',
    'Afrikaans': 'af',
}


class MultilingualTranslator:
    """Translator with caching and multiple backend support"""
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize translator
        
        Args:
            cache_size: Size of translation cache
        """
        self.cache_size = cache_size
        self.supported_languages = LANGUAGE_CODES
        
        # Test connection
        try:
            test_translator = GoogleTranslator(source='en', target='es')
            test_translator.translate('test')
            logger.info("Translation service initialized successfully")
        except Exception as e:
            logger.warning(f"Translation service initialization warning: {e}")
    
    @lru_cache(maxsize=1000)
    def translate(self, text: str, target_language: str, source_language: str = 'en') -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code or name
            source_language: Source language code (default: 'en')
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        # Convert language name to code if needed
        if target_language in self.supported_languages:
            target_code = self.supported_languages[target_language]
        else:
            target_code = target_language
        
        # No translation needed if same language
        if source_language == target_code:
            return text
        
        try:
            translator = GoogleTranslator(source=source_language, target=target_code)
            translated = translator.translate(text)
            logger.info(f"Translated '{text}' from {source_language} to {target_code}: '{translated}'")
            return translated
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def batch_translate(self, texts: List[str], target_language: str, 
                       source_language: str = 'en') -> List[str]:
        """
        Translate multiple texts
        
        Args:
            texts: List of texts to translate
            target_language: Target language code or name
            source_language: Source language code
            
        Returns:
            List of translated texts
        """
        return [self.translate(text, target_language, source_language) for text in texts]
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages"""
        return self.supported_languages
    
    def get_language_names(self) -> List[str]:
        """Get list of supported language names"""
        return list(self.supported_languages.keys())
    
    def get_language_code(self, language_name: str) -> Optional[str]:
        """Get language code from language name"""
        return self.supported_languages.get(language_name)


class TextToSpeech:
    """Text-to-speech converter"""
    
    def __init__(self):
        """Initialize TTS"""
        self.temp_dir = tempfile.gettempdir()
    
    def generate_speech(self, text: str, language_code: str = 'en', 
                       slow: bool = False) -> Optional[str]:
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            language_code: Language code for TTS
            slow: Whether to speak slowly
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        try:
            # Create TTS object
            tts = gTTS(text=text, lang=language_code, slow=slow)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.mp3',
                dir=self.temp_dir
            )
            tts.save(temp_file.name)
            
            logger.info(f"✅ Generated speech for text: '{text}' in language: {language_code}")
            logger.info(f"   Audio saved to: {temp_file.name}")
            return temp_file.name
        
        except Exception as e:
            logger.error(f"❌ TTS error for text '{text}' in language '{language_code}': {e}")
            
            # Try fallback to English if original language failed
            if language_code != 'en':
                try:
                    logger.info("Attempting fallback to English TTS...")
                    tts = gTTS(text=text, lang='en', slow=slow)
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix='.mp3',
                        dir=self.temp_dir
                    )
                    tts.save(temp_file.name)
                    logger.info(f"✅ Fallback English TTS successful")
                    return temp_file.name
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback TTS also failed: {fallback_error}")
            
            return None


class SignLanguageTranslationPipeline:
    """Complete translation pipeline for sign language"""
    
    def __init__(self, enable_tts: bool = True):
        """
        Initialize translation pipeline
        
        Args:
            enable_tts: Whether to enable text-to-speech
        """
        self.translator = MultilingualTranslator()
        self.tts = TextToSpeech() if enable_tts else None
        self.enable_tts = enable_tts
    
    def process(self, 
                english_text: str, 
                target_language: str = 'English',
                generate_audio: bool = True) -> Dict[str, any]:
        """
        Process sign language prediction through translation pipeline
        
        Args:
            english_text: Predicted English text from model
            target_language: Target language for translation
            generate_audio: Whether to generate audio output
            
        Returns:
            Dictionary with translation results
        """
        result = {
            'english_text': english_text,
            'target_language': target_language,
            'translated_text': None,
            'audio_path': None,
            'language_code': None
        }
        
        # Get language code
        language_code = self.translator.get_language_code(target_language)
        if not language_code:
            language_code = target_language  # Assume it's already a code
        
        result['language_code'] = language_code
        
        # Translate
        if target_language != 'English' and language_code != 'en':
            translated_text = self.translator.translate(
                english_text, 
                target_language=language_code,
                source_language='en'
            )
            result['translated_text'] = translated_text
        else:
            result['translated_text'] = english_text
        
        # Generate audio
        if generate_audio and self.enable_tts and self.tts:
            logger.info(f"🔊 Generating audio for: '{result['translated_text']}' in {language_code}")
            audio_path = self.tts.generate_speech(
                result['translated_text'],
                language_code=language_code
            )
            result['audio_path'] = audio_path
            
            if audio_path:
                logger.info(f"✅ Audio generation successful")
            else:
                logger.warning(f"⚠️ Audio generation failed for language: {language_code}")
        else:
            if not generate_audio:
                logger.info("🔇 Audio generation disabled by user")
            elif not self.enable_tts:
                logger.warning("⚠️ TTS not enabled in pipeline")
        
        return result
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.translator.get_language_names()


# Convenience functions
def translate_text(text: str, target_language: str, source_language: str = 'en') -> str:
    """Quick translation function"""
    translator = MultilingualTranslator()
    return translator.translate(text, target_language, source_language)


def generate_speech(text: str, language_code: str = 'en') -> Optional[str]:
    """Quick TTS function"""
    tts = TextToSpeech()
    return tts.generate_speech(text, language_code)
