def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f"<style>{f.read()}</style>"