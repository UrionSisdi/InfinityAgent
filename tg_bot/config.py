from pydantic import PostgresDsn

class Settings:
    # Telegram settings
    TELEGRAM_TOKEN: str = "7857311489:AAHo_jO_rBW9xaD8k-jXxEBFiiE2VLD9NsQ"
    
    # PostgreSQL settings - используем жестко заданные значения
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "infinity_agent"
    
    @property
    def postgres_dsn(self) -> PostgresDsn:
        print(f"В методе postgres_dsn, POSTGRES_DB: {self.POSTGRES_DB}")
        return PostgresDsn.build(
            scheme="postgresql",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=int(self.POSTGRES_PORT),
            path=f"{self.POSTGRES_DB}",  # Добавляем слеш в начале пути
        )


settings = Settings()
print(f"Значение settings.POSTGRES_DB: {settings.POSTGRES_DB}") 