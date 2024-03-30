import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import datetime
import signal
import sys

# Need to install reqeusts and pandas for this to work!!!
# Authors: Akshat Jain, Christen Xie, Ojas Vashishta, Liam Manatt

# Import dataset and create dataframe
songs_df = pd.read_csv(
    "/Users/akshat/Documents/cse151aproject/spotifydata.csv", encoding="ISO-8859-1"
)
songs_df_updated = pd.DataFrame(columns=songs_df.columns.tolist() + ["popularity"])

# Hidden for github
client_id = ""
client_secret = ""

# Get access token from spotify


def get_access_token(client_id, client_secret):
    url = "https://accounts.spotify.com/api/token"
    response = requests.post(
        url,
        data={"grant_type": "client_credentials"},
        auth=HTTPBasicAuth(client_id, client_secret),
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception("couldn't find token")


# Get track popularity and print to console that it was receved correctly
# Was useful when we had too many requests and had to switch to new client id and secret


def get_track_popularity(track_id, access_token, index, total):
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        popularity = response.json()["popularity"]
        print(f"[{index}/{total}] found {popularity} for id {track_id}")
        return popularity
    else:
        print(
            response.content  # Usually ended up being an issue with too many requests
        )
        print(f"[{index}/{total}] failed to find popularity for {track_id}")
        return None


# If script ends early (when too many reqeusts and spotify keeps denying requets)
# Save how many songs were fetched and current time into name of the csv


def signal_handler(sig, frame):
    global songs_df_updated
    if not songs_df_updated.empty:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{len(songs_df_updated)}_songs_updated_{now}.csv"
        songs_df_updated.to_csv(filename, index=False, encoding="utf-8-sig")
        print(
            f"\nCSV file {filename} has been saved with {len(songs_df_updated)} entries."
        )
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Start at song 0 by default


def main(start=0):  # Add start parameter with default value 0
    access_token = get_access_token(client_id, client_secret)
    total_songs = len(songs_df)

    for index, row in songs_df.iloc[start:].iterrows():
        popularity = get_track_popularity(
            row["id"], access_token, index + 1, total_songs
        )
        if popularity is not None:
            songs_df_updated.loc[len(songs_df_updated)] = row.tolist() + [popularity]

    # This is assuming the script makes it to end without any interruptions
    signal_handler(None, None)


if __name__ == "__main__":
    start_line = 9658  # Last time spotify said too many requests was at 9657 songs, so we switched client id and secret and started at this row
    main(start=start_line)
