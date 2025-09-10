from typing import Optional

from huggingface_hub import HfApi


def publish_folder_to_hub(
    folder: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Add model artifacts via autopack",
    revision: Optional[str] = None,
    create: bool = True,
) -> str:
    """Publish a local folder to the Hugging Face Hub repo.

    Returns the URL to the uploaded repository.
    """
    api = HfApi()
    if create:
        api.create_repo(repo_id=repo_id, token=token, repo_type="model", private=private, exist_ok=True)

    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        path_in_repo=".",
        commit_message=commit_message,
        revision=revision,
        token=token,
        create_pr=False,
        run_as_future=False,
    )
    return result.repo_url


