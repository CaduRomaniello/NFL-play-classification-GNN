from datetime import datetime

# crie uma classe Logger que terá vários métodos estáticos para registrar mensagens de log em diferentes níveis (info, warning, error).
class Logger:
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    @staticmethod
    def info(message=""):
        if not message:
            print()
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Logger.BLUE}[INFO]{Logger.RESET} {now} - {message}")

    @staticmethod
    def warning(message=""):
        if not message:
            print()
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Logger.YELLOW}[WARNING]{Logger.RESET} {now} - {message}")

    @staticmethod
    def error(message=""):
        if not message:
            print()
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Logger.RED}[ERROR]{Logger.RESET} {now} - {message}")

    @staticmethod
    def success(message=""):
        if not message:
            print()
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Logger.GREEN}[SUCCESS]{Logger.RESET} {now} - {message}")