import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime

client_credentials_manager = SpotifyClientCredentials(client_id='afc39e8b75f64a12a6d050db288a6653', client_secret='c751f8fb694a4099b88b4c4e160466bf')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_playlist_tracks(playlist_uri):
    tracks = []
    offset = 0
    print(playlist_uri)
    while True:
        response = sp.playlist_tracks(playlist_uri, offset=offset)
        if not response['items']:
            break
        tracks.extend(response['items'])
        offset += len(response['items'])
    return tracks

# Function to retrieve audio features for a batch of track IDs
def get_audio_features_batch(track_ids):
    features = []
    for i in range(0, len(track_ids), 90):  # Process 90 IDs at a time
        features.extend(sp.audio_features(track_ids[i:i+90]))
    return features

# List of playlist URIs
playlist_URIs = [
    "https://open.spotify.com/playlist/3bQy66sMaRDIUIsS7UQnuO",
    "https://open.spotify.com/playlist/4YOfhHpjPB0tq29NPpDY3F",
    "https://open.spotify.com/playlist/5cwtgqs4L1fX8IKoQebfjJ",
    "https://open.spotify.com/playlist/37i9dQZF1DWX76Z8XDsZzF",
    "https://open.spotify.com/playlist/6mfCulwURJTYzcAo7vvgoM",
    "https://open.spotify.com/playlist/37i9dQZF1DWTwbZHrJRIgD",
    "https://open.spotify.com/playlist/3VwENabqmIlGIRXQltdYcG",
    "https://open.spotify.com/playlist/5O7hB1FhIcs6xShOV9AIIz",
    "https://open.spotify.com/playlist/5JVuHeTFLMdeoa8DmPY3jL",
    "https://open.spotify.com/playlist/3AAeD7XiPsK3UtmXyPCvfz",
    "https://open.spotify.com/playlist/37i9dQZF1DXdFesNN9TzXT",
    "https://open.spotify.com/playlist/3YfKWpfw1nROFLbyaFQwdP",
    "https://open.spotify.com/playlist/6ca0Lg3gXmzwqEh79TjK7L",
    "https://open.spotify.com/playlist/6WiMSiEX3FleOOQimEKzVQ",
    "https://open.spotify.com/playlist/6aH0SsaVm9ysgJxS4h5bNH",
    "https://open.spotify.com/playlist/2KAl1ayr9hJLXbic137j4W",
    "https://open.spotify.com/playlist/37i9dQZF1DWX3SoTqhs2rq",
    "https://open.spotify.com/playlist/5AiPzkrvxDguoJy9CR0OdF",
    "https://open.spotify.com/playlist/3Wek9RlkXhelSK3oidgHTS",
    "https://open.spotify.com/playlist/1QbJym8B3Xb6rpXdk7MBiJ",
    "https://open.spotify.com/playlist/3JNWpteYvH3ynMcyPcvxfx",
]

# Dictionary mapping playlist URIs to corresponding emotions
playlist_emotions = {
    "https://open.spotify.com/playlist/3bQy66sMaRDIUIsS7UQnuO": "happy",
    "https://open.spotify.com/playlist/4YOfhHpjPB0tq29NPpDY3F": "sad",
    "https://open.spotify.com/playlist/5cwtgqs4L1fX8IKoQebfjJ": "angry",
    "https://open.spotify.com/playlist/37i9dQZF1DWX76Z8XDsZzF":"chill",
    "https://open.spotify.com/playlist/6mfCulwURJTYzcAo7vvgoM": "sad",
    "https://open.spotify.com/playlist/37i9dQZF1DWTwbZHrJRIgD": "happy",
    "https://open.spotify.com/playlist/3VwENabqmIlGIRXQltdYcG":"soft",
    "https://open.spotify.com/playlist/5O7hB1FhIcs6xShOV9AIIz":"angry",
    "https://open.spotify.com/playlist/5JVuHeTFLMdeoa8DmPY3jL":"sad",
    "https://open.spotify.com/playlist/3AAeD7XiPsK3UtmXyPCvfz": "soft",
    "https://open.spotify.com/playlist/37i9dQZF1DXdFesNN9TzXT": "sad",
    "https://open.spotify.com/playlist/3YfKWpfw1nROFLbyaFQwdP":"chill",
    "https://open.spotify.com/playlist/6ca0Lg3gXmzwqEh79TjK7L": "angry",
    "https://open.spotify.com/playlist/6WiMSiEX3FleOOQimEKzVQ":"happy",
    "https://open.spotify.com/playlist/6aH0SsaVm9ysgJxS4h5bNH":"soft",
    "https://open.spotify.com/playlist/2KAl1ayr9hJLXbic137j4W": "chill",
    "https://open.spotify.com/playlist/37i9dQZF1DWX3SoTqhs2rq": "happy",
    "https://open.spotify.com/playlist/5AiPzkrvxDguoJy9CR0OdF":"angry",
    "https://open.spotify.com/playlist/3Wek9RlkXhelSK3oidgHTS": "sad",
    "https://open.spotify.com/playlist/1QbJym8B3Xb6rpXdk7MBiJ":"chill",
    "https://open.spotify.com/playlist/3JNWpteYvH3ynMcyPcvxfx":"angry"
}

all_data = []  # List to hold data from all playlists

for playlist_URI in playlist_URIs:
    songs = get_playlist_tracks(playlist_URI)
    track_ids = [song['track']['id'] for song in songs if song['track'] is not None and song['track']['id'] is not None]
    song_name = []
    song_popu = []
    song_added_date = []
    song_release_date = []
    artists_col = []
    for song in songs:
        if song['track'] is not None:
            song_name.append(song['track']['name'])
            song_popu.append(song['track']['popularity'])
            song_added_date.append(song['added_at'])
            song_release_date.append(song['track']['album']['release_date'])
            all_artists = song['track']['artists']
            artists = []
            for a in all_artists:
                artists.append(a['name'])
            artists_col.append(artists)

    df = pd.DataFrame({
        'name': song_name,
        'popularity': song_popu,
        'date_added': pd.to_datetime(song_added_date),
        # 'release_year': list(map(lambda x: int(x[:4]), song_release_date)),
        'release_year':[int(x[:4]) if x is not None else None for x in song_release_date],
        'artists': artists_col
    })

    # Audio features
    features = get_audio_features_batch(track_ids)
    feat_names = list(sp.audio_features(track_ids[0])[0].keys())

    for row in range(len(features)):
        for col in range(len(feat_names)):
            df.loc[row, feat_names[col]] = features[row][feat_names[col]]

    # Add emotion column
    df['song_emotion'] = playlist_emotions[playlist_URI]

    all_data.append(df)

# Concatenate data from all playlists
final_df = pd.concat(all_data, ignore_index=True)

# Save DataFrame to CSV file
final_df.to_csv('spotify_playlist_all_emotions_songs_latest.csv', index=False)

print("DataFrame saved to 'spotify_playlist_all_emotions_songs.csv' successfully.")