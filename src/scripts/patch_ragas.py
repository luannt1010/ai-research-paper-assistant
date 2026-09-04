from importlib.util import find_spec
from pathlib import Path


spec = find_spec("ragas")

if spec is None:
    raise RuntimeError("RAGAS is not installed.")

ragas_dir = Path(
    list(spec.submodule_search_locations)[0]
)

base_file = ragas_dir / "llms" / "base.py"

content = base_file.read_text(
    encoding="utf-8"
)

content = content.replace(
    "from langchain_community.chat_models.vertexai "
    "import ChatVertexAI",
    "from langchain_google_vertexai "
    "import ChatVertexAI",
)

content = content.replace(
    "from langchain_community.llms import VertexAI",
    "from langchain_google_vertexai "
    "import VertexAI",
)

base_file.write_text(
    content,
    encoding="utf-8"
)

print(f"RAGAS patched successfully: {base_file}")