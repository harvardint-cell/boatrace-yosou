"""site/ を gh-pages ブランチに上書きpushする(履歴は1コミットだけ保持)。
ローカルで実行した場合(GITHUB_ACTIONS未設定)は何もしない。"""
import os
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def publish():
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    remote = subprocess.check_output(["git", "-C", str(ROOT), "remote", "get-url", "origin"], text=True).strip()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(ROOT / "site", tmp, dirs_exist_ok=True)
        pathlib.Path(tmp, ".nojekyll").touch()
        run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)  # noqa: E731
        run("init", "-q", "-b", "gh-pages")
        run("config", "user.name", "github-actions[bot]")
        run("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        run("add", "-A")
        run("commit", "-qm", "update predictions")
        for _ in range(3):
            r = subprocess.run(["git", "-C", tmp, "push", "-qf", remote, "gh-pages"], capture_output=True)
            if r.returncode == 0:
                return
        print("publish failed", r.stderr.decode()[-300:], flush=True)
