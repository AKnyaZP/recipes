"""
ASR модуль с T-one от Т-Банка (исправлены warnings).
"""

import torch
import numpy as np
from typing import Dict, Any
import logging
import io
import warnings

# Подавляем warning о torch_dtype
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToneASR:
    """
    T-one ASR от Т-Банка (70M параметров).
    HuggingFace: https://huggingface.co/t-tech/T-one
    """

    def __init__(self, device: str = "auto", max_memory_usage: float = 0.9):
        """
        Args:
            device: "cpu", "cuda" или "auto"
            max_memory_usage: Процент использования GPU памяти (0.9 = 90%)
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_memory_usage = max_memory_usage
        self.model = None
        self.processor = None

        self._init_model()

    def _init_model(self):
        """Загружает T-one с правильными параметрами."""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import warnings

            logger.info("📦 Загружаем T-one (70M параметров)...")

            model_name = "t-tech/T-one"

            # Загружаем processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)

            # Параметры загрузки модели
            model_kwargs = {
                "low_cpu_mem_usage": True,
            }

            # Настройки для GPU
            if self.device == "cuda":
                # Используем dtype вместо torch_dtype
                model_kwargs["torch_dtype"] = torch.float16

                # Настройки памяти для accelerate
                max_memory = {0: f"{int(self.max_memory_usage * 100)}%"}
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = max_memory

            # Подавляем deprecation warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*torch_dtype.*")

                self.model = Wav2Vec2ForCTC.from_pretrained(
                    model_name,
                    **model_kwargs
                )

            # Для CPU переносим модель вручную
            if self.device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()

            logger.info(f"✅ T-one загружена на {self.device}")

            # Показываем использование памяти для GPU
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"📊 Использовано GPU памяти: {allocated:.2f} GB")

        except ImportError as e:
            raise ImportError(
                "Установите зависимости:\n"
                "pip install transformers torch torchaudio soundfile librosa accelerate"
            ) from e
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
            raise

    def transcribe_bytes(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Распознаёт речь из байтов.

        Args:
            audio_data: Аудио в байтах (WAV формат)
            sample_rate: Частота дискретизации

        Returns:
            {"text": "распознанный текст", "success": True}
        """
        try:
            import soundfile as sf

            # Читаем WAV из байтов
            audio_io = io.BytesIO(audio_data)
            waveform, sr = sf.read(audio_io)

            # Преобразуем в numpy array
            if isinstance(waveform, np.ndarray):
                if waveform.ndim > 1:
                    waveform = waveform[:, 0]  # Моно

            # Ресэмплируем если нужно
            if sr != 16000:
                try:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                except ImportError:
                    # Fallback: используем torchaudio
                    import torchaudio
                    waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform_tensor).squeeze().numpy()

            # Распознаём
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            # Переносим на device
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Инференс
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Декодируем
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            text = transcription.strip()

            logger.info(f"🎤 Распознано: {text}")

            return {
                "text": text,
                "success": True,
                "engine": "t-one",
                "language": "ru"
            }

        except Exception as e:
            logger.error(f"❌ Ошибка распознавания: {e}")
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "engine": "t-one"
            }

    def get_info(self) -> Dict[str, str]:
        """Информация о модели."""
        info = {
            "engine": "T-one (Т-Банк)",
            "status": "✅ Готов" if self.model else "❌ Не загружен",
            "parameters": "70M",
            "device": self.device,
            "github": "https://github.com/voicekit-team/T-one"
        }

        # Добавляем информацию о памяти для GPU
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["gpu_memory"] = f"{allocated:.2f}GB / {total:.2f}GB"

        return info
