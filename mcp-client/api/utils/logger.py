import logging
import sys
import io
import platform

# Настраиваем кодировку для Windows
is_windows = platform.system() == "Windows"

# Класс для безопасного логирования с обработкой ошибок кодировки
class SafeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Безопасная запись с обработкой ошибок кодировки
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # При ошибке кодировки (например, с эмодзи в Windows)
                # Заменяем проблемные символы или используем UTF-8
                safe_msg = msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging
logger = logging.getLogger("MCPClient")
logger.setLevel(logging.DEBUG)

# File handler with DEBUG level
file_handler = logging.FileHandler("mcp_client.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Console handler with INFO level
console_handler = SafeStreamHandler(sys.stdout) if is_windows else logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)
