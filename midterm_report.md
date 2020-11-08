# Midterm Report
### Megan Yin (my474), Fionna Chen (fyc3)

## The Data

### Tables
The Spotify dataset contains 5 tables on song data:
1. `data.csv` contains more than 160,000 songs collected from the [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
2. `data_by_artist.csv` groups this information in `data.csv` by artist
3. `data_by_genres.csv` groups by genres
4. `data_by_year` groups by year
5. `data_w_genres.csv` adds the genre information for each song

### Features

In this dataset we have 19 features and 169909 examples. We have numerical, boolean, and categorical features described below.

<small> Descriptions based on [Audio Features](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/) in [Spotify Web API](https://developer.spotify.com/documentation/web-api/) </small>

#### Numerical
* acousticness (0.0-1.0): A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. 
* danceability (0.0-1.0): describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
* energy (0.0-1.0): a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. 
* duration_ms (int): duration of track in milliseconds
* instrumentalness (0.0-1.0): Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. _Values above 0.5 are intended to represent instrumental tracks_, but confidence is higher as the value approaches 1.0.
* valence (0.0-1.0): A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). 
* popularity 
* tempo (float): The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
* liveness (0.0-1.0): Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
* loudness (float): The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
* Speechiness (float): detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
* year (1921-2020): year the song was released

#### Boolean
* mode: 0 = minor, 1 = major
* explicit: 0 = no explicit content, 1 = explicit content

#### Categorical
* key (int): The estimated overall key of the track. Integers map to pitches using standard [Pitch Class notation](https://en.wikipedia.org/wiki/Pitch_class). E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
* artists (string[]): list of artists on the track
* release_date (yyyy-mm-dd): release date of the track (precision varies)
* name (string): name of the song

### Histograms
We wanted to get a sense of the distribution of our data on some of these (numerical) features so we used the pandas `Dataframe.hist` function to plot histograms of key features.

<img src="./imgs/data-histograms.png" />

Observations:
* acousticness: we usually have high confidence on whether a track is acoustic or not. (Most tracks are in the 0-0.1 or 0.9-1.0 range of acousticness). If these acousticness scores are accurate, we have about an equal distribution of acoustic vs. not acoustic tracks
* danceability: this metric is pretty normally distributed across songs (but perhaps with a heavier left tail). Most songs are moderately danceable (mid-valued), with very few songs on the extremes.
* energy: the data is relatively equally distributed on energy values
* instrumentalness: We mostly have non-instrumental songs since the histogram shows most songs around 0.0-0.1 instrumentalness.
* key: we have a good distribution of keys with relatively more of them in the 0th key of C
* liveness: most of our songs are not live (don't have an audience component).
* loudness: most songs are not loud with a loudness value close to 0. However, we do have some songs that are very loud. We have very few in the mid-range, though.
* popularity: most songs have popularities in 0-80 range with relatively higher density in 0-10 range meaning very unpopular. This makes sense because relatively few songs are very popular; most songs are niche and unpopular.
* speechiness: Most of our songs are not speechy, although we do have some speechy ones that may be raps or something. We can probably check the very speechy ones to make sure they are still songs and not audio files of a talk show or podcast.
* tempo: We don't have any very slow songs (tempos are all > 50). Tempos are normally distributed with some very fast songs.
* valence: also rather normally distributed which means we have a variety of moods present in the dataset.
* year: we consistently have 2000 songs for every year after around 1950. Before then, the number of songs for each year was increasing. 

## Missing Values
We're lucky to not have any missing values in this dataset. This is likely because the data comes from the Spotify API directly and we limit ourselves to only songs from there. Thus any songs Spotify has would have all of these features defined and outputted in the API.

## Feature Transformations
The `artist` feature column is a list of artists on that track. The elements of that list are strings with the names of the artist. We can encode this in a many hot encoding such that a $1$ in column $i$ means artist $i$ is featured on that track. This can be interesting for us to predict popularity or see how certain artists have gotten more popular over time. Certainly, we have seen the artist has a huge effect on the popularity of a song which is why smaller artists try to work on songs with more popular artists.

Most other features have already been scaled and normalized for us on a 0-1 scale fortunately so we should be fine to use the raw data. Perhaps to reduce noise, we may consider changing some of these values to boolean. This may also help overfitting. 

## Over and Underfitting
If we were to just use the 19 features as is we run a risk of underfitting the model since there are just too few features to capture the 160k+ examples. However, we will expand the artists feature to a many-hot encoded vector representing artists on the track. This will drastically increase the number of features, but this is still a sparse matrix. 

Other ways to avoid underfitting is by fitting a more complex model. For example, we can experiment with various forms of Polynomial Regression or random forest/SVM regressions.

To prevent overfitting, we should separate our data into the 80/20 train test set split. We will also look into applying 10-fold cross validation. 

## Naive Regressions
We wanted to get a sense of how our model would perform if we just trained models on

