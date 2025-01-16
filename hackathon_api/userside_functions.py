import aiohttp
import pandas as pd
from torch import Tensor

USERNAME: str = "admin"
PASSWORD: str = "test123"
BASE_URL = "http://127.0.0.1:8000"


async def register(username: str = USERNAME, passwd: str = PASSWORD) -> None:
    """
    Calls the /register endpoint to create a new user.
    Raises ValueError if registration fails.
    """
    endpoint = f"{BASE_URL}/register?username={username}&password={passwd}"
    async with aiohttp.ClientSession() as session:
        res = await session.post(endpoint)  # (no JSON body here)
        if res.status != 200:
            content = await res.text()
            raise ValueError(f"Registration failed: {content}")
        print(await res.json())


async def get_leaderboard() -> pd.DataFrame:
    """
    Calls the /leaderboard endpoint and returns a DataFrame
    with columns [user, score].
    """
    endpoint = f"{BASE_URL}/leaderboard"
    async with aiohttp.ClientSession() as session:
        res = await session.get(endpoint)
        if res.status != 200:
            content = await res.text()
            raise ValueError(f"Error getting leaderboard: {content}")
        data = await res.json()
        leaderboard = data["leaderboard"]
        df = pd.DataFrame(leaderboard, columns=["user", "score"])
        return df


async def get_submission_history(username: str = USERNAME) -> list[float]:
    """
    Calls the /submissions_history endpoint with the given username
    and returns that user's history of submission scores as a list[float].
    """
    endpoint = f"{BASE_URL}/submissions_history"

    async with aiohttp.ClientSession() as session:
        # Pass the username as a query parameter
        res = await session.get(endpoint, params={"username": username})
        if res.status != 200:
            content = await res.text()
            raise ValueError(f"Error getting submissions history: {content}")
        data = await res.json()
        return data["history"]


async def submit(
    perturbations: list[Tensor], username: str = USERNAME, passwd: str = PASSWORD
) -> float:
    """
    Calls the /submit endpoint (requires Basic Auth).
    Uploads each tensor in the `perturbations` list as an image file
    in a multipart/form-data request. Returns the score from the server.
    """

    if not isinstance(perturbations, list) or not all(
        isinstance(pert, Tensor) for pert in perturbations
    ):
        raise ValueError(
            "Pertubations have wrong type, you need to pass list of torch.Tensor"
        )

    endpoint = f"{BASE_URL}/submit"
    form_data = aiohttp.FormData()

    for i, tensor in enumerate(perturbations):
        file_bytes = tensor.cpu().numpy().tobytes()
        form_data.add_field(
            name="files",
            value=file_bytes,
            filename=f"file_{i}.bin",
            content_type="application/octet-stream",
        )

    # Send request with BasicAuth
    async with aiohttp.ClientSession() as session:
        res = await session.post(
            endpoint, data=form_data, auth=aiohttp.BasicAuth(username, passwd)
        )
        if res.status != 200:
            content = await res.text()
            raise ValueError(f"Error submitting perturbations: {content}")
        data = await res.json()
        return data["score"]
