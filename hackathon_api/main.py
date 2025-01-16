import datetime
import typing as tp
from collections import defaultdict

import torch
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from torch import Tensor
from torchvision import transforms

from hackathon_api.adversarial_attack_scoring import (
    check_validity_of_pertubations,
    score_submission,
)

app = FastAPI()
security = HTTPBasic()

DIMENSIONS = (3, 32, 32)
MAX_ATTACKS_PER_SUBMIT = 2


SUBMISSIONS_HISTORY: dict[str, list[float]] = defaultdict(list)
USERS_DB = {}
LEADERBOARD: dict[str, float] = defaultdict(float)

DEADLINE = datetime.datetime(2025, 1, 20, 16, 30)

cast_to_tensor: tp.Callable[[Image.Image], Tensor] = transforms.ToTensor()


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Check if the user is valid; if not, raise an HTTP 401.
    """
    username = credentials.username
    password = credentials.password

    # Validate the user
    if username not in USERS_DB or USERS_DB[username] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


@app.post("/register")
def register_user(username: str, password: str):
    """
    Register a new user with username and password.
    """
    if username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    USERS_DB[username] = password
    return {"message": f"User '{username}' registered successfully!"}


@app.post("/submit")
async def submit_adversarial_perturbation(
    files: list[UploadFile] = File(...),
    current_user: str = Depends(get_current_user),
) -> dict[str, str | float]:
    """
    Accept multiple uploaded image files from authenticated user.
    Convert each to a 3-channel tensor (RGB) and compute a score.
    Keep only the highest score for each user.
    """

    if datetime.datetime.now() > DEADLINE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Deadline is due, system does not accept submissions anymore!"
                "Thanks for your participation"
            ),
        )

    if len(files) > MAX_ATTACKS_PER_SUBMIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many attack submitted. Only {MAX_ATTACKS_PER_SUBMIT} attacks per submission are allowed",
        )

    tensors = []
    for file in files:
        file_bytes = await file.read()
        tensor = torch.frombuffer(file_bytes, dtype=torch.float32)

        try:
            tensor = tensor.reshape(DIMENSIONS)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Seems that your pertubations have wrong shape, (3,32,32) expected",
            )

        tensors.append(tensor)

    if not check_validity_of_pertubations(tensors):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pertubations are too big, check their norm",
        )

    # Score the submission
    score = score_submission(tensors)

    # Store only the highest score
    LEADERBOARD[current_user] = max(score, LEADERBOARD[current_user])
    SUBMISSIONS_HISTORY[current_user].append(score)
    return {"user": current_user, "score": score}


@app.get("/leaderboard")
def get_leaderboard() -> dict[str, list[tuple[str, float]]]:
    """
    Return the sorted leaderboard (descending order by score).
    """
    sorted_leaderboard = sorted(LEADERBOARD.items(), key=lambda x: x[1], reverse=True)
    leaderboard_list = [(user, score) for user, score in sorted_leaderboard]

    return {"leaderboard": leaderboard_list}


@app.get("/submissions_history")
def get_submissions_history(username: str) -> dict[str, list[float]]:
    """
    Return the sorted leaderboard (descending order by score).
    """

    return {"history": SUBMISSIONS_HISTORY[username]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
