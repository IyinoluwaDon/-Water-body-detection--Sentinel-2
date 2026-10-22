"""Refresh the self-contained Colab payload from maintained source, with deterministic zip bytes."""
import base64
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def payload():
    buffer = io.BytesIO()
    paths = [ROOT / p for p in ('pyproject.toml', 'README.md', 'config.yaml')]
    paths += sorted((ROOT / 'src/waterdet').glob('*.py'))
    paths += sorted((ROOT / 'docs').glob('*.md'))
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    path = ROOT / 'notebooks/waterdet_colab.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if 'PACKAGE_ZIP_B64 = ' in source:
            source = re.sub(r'PACKAGE_ZIP_B64 = "[^"]*"', f'PACKAGE_ZIP_B64 = "{payload()}"', source)
            # Refresh code on every run while preserving the user's configuration and data.
            marker = 'if not (work / "config.yaml").exists():'
            start = source.index(marker if marker in source else 'work.mkdir(parents=True, exist_ok=True)')
            end = source.index('os.chdir(work)', start)
            source = source[:start] + '''work.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PACKAGE_ZIP_B64))) as bundle:
    for member in bundle.namelist():
        if member == "config.yaml" and (work / member).exists():
            continue
        bundle.extract(member, work)
''' + source[end:]
        cell['source'] = source.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
