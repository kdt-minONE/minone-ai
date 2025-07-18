from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FILE_MAX_SIZE: int = 10485760  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True

    # AI 설정
    OPENAI_API_KEY: Optional[str] = None

    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API 키 접근자 (snake_case)"""
        return self.OPENAI_API_KEY

    TAVILY_API_KEY: Optional[str] = None

    # CORS 설정
    ALLOWED_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
