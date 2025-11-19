# Movie Mixer 🎬

A hybrid movie recommendation system that suggests films based on two input movies using TF-IDF vectorization and cosine similarity. Built with Streamlit for an interactive web interface.

## Overview

**Movie Mixer** combines the characteristics of two movies you love to find the perfect recommendations. By analyzing genres and keywords, it creates a "hybrid vector" that captures the essence of both films and suggests movies with similar attributes.

### How It Works

1. **Select Two Movies**: Choose any two movies from our curated database of popular films
2. **Hybrid Vector Creation**: The system extracts TF-IDF features from both movies and creates a soft intersection using element-wise minimum
3. **Similarity Matching**: Computes cosine similarity between the hybrid vector and all other movies
4. **Smart Recommendations**: Returns the top 3 most similar movies, complete with posters, ratings, and descriptions

## Features

✨ **Intelligent Recommendations**: Uses machine learning (TF-IDF + cosine similarity) for accurate suggestions

🎯 **Hybrid Approach**: Combines features from two movies to find unique matches

🖼️ **Rich UI**: Beautiful movie cards with posters fetched from TMDB API

⚡ **High Performance**: Cached data loading and model fitting for instant results

🎨 **Clean Architecture**: Modular codebase with separation of concerns

📊 **Quality Filtering**: Only includes movies with 1000+ votes for reliable recommendations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Lordiod/Movie-Mixer.git
   cd Movie-Mixer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your TMDB API token:
   ```env
   TMDB_TOKEN=your_tmdb_bearer_token_here
   ```
   
   To get a TMDB API token:
   - Create a free account at [TMDB](https://www.themoviedb.org/)
   - Go to Settings → API
   - Generate an API Read Access Token (v4 auth)
   - Copy the Bearer token to your `.env` file

4. **Verify data files**
   
   Ensure the following CSV files are present in the `assets/` directory:
   - `movies.csv` - Movie metadata including titles, genres, ratings
   - `keywords.csv` - Movie keywords for enhanced recommendations

## Usage

### Running the Application

Launch the Streamlit app with:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Select your first movie** from the dropdown menu
2. **Select your second movie** from the second dropdown
3. View the **selected movies** with their posters and details
4. Click **"Get Recommendations"** to see suggested movies
5. Explore the **recommendations** with full details and posters

### Example

Try mixing:
- "The Matrix" + "Inception" → Mind-bending sci-fi thrillers
- "Toy Story" + "Finding Nemo" → Heartwarming Pixar adventures
- "The Dark Knight" + "Iron Man" → Epic superhero films

## Project Structure

```
Movie-Mixer/
│
│── app.py                   # Streamlit entry point with minimal logic
│── requirements.txt         # Python dependencies
│── README.md               # Project documentation (this file)
│
│── src/                    # Source code modules
│    ├── __init__.py       # Package initialization
│    ├── data_loader.py    # CSV loading and data cleaning
│    ├── preprocess.py     # Genre/keyword extraction and feature engineering
│    ├── model.py          # TF-IDF vectorization and recommendation engine
│    ├── tmdb_api.py       # TMDB API integration for posters
│    └── interface.py      # Streamlit UI components
│
│── assets/                # Data files
│    ├── movies.csv       # Movie metadata
│    └── keywords.csv     # Movie keywords
│
└── .gitignore            # Git ignore patterns
```

### Module Descriptions

- **`app.py`**: Main entry point that orchestrates the application flow
- **`data_loader.py`**: Handles CSV file loading with encoding fallbacks and ID cleaning
- **`preprocess.py`**: Cleans genre/keyword JSON strings and creates combined feature fields
- **`model.py`**: Implements the `MovieRecommender` class with TF-IDF and cosine similarity
- **`tmdb_api.py`**: Interfaces with TMDB API to fetch movie posters
- **`interface.py`**: Contains all Streamlit UI rendering functions

## Technical Details

### Algorithm

The recommendation system uses a **hybrid vector approach**:

1. **Feature Extraction**: Extracts genres and keywords from both input movies
2. **TF-IDF Vectorization**: Converts text features into numerical vectors
3. **Hybrid Vector**: Computes element-wise minimum of both movie vectors (soft intersection)
4. **Similarity Scoring**: Calculates cosine similarity between hybrid vector and all movies
5. **Ranking**: Returns top N movies with highest similarity scores

### Technologies

- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: TF-IDF vectorization and cosine similarity
- **Requests**: HTTP library for TMDB API calls
- **NumPy**: Numerical computing for vector operations

### Performance Optimizations

- `@st.cache_data` for data loading and preprocessing
- `@st.cache_resource` for model initialization
- Efficient sparse matrix operations with scikit-learn
- Pre-computed cosine similarity matrix

## Data Sources

- **Movie Data**: The Movies Dataset from [TMDB](https://www.themoviedb.org/)
- **Posters**: Fetched in real-time from TMDB API

## Configuration

### Environment Variables

The application uses environment variables for sensitive configuration. All settings are stored in the `.env` file:

- **`TMDB_TOKEN`**: Your TMDB API Read Access Token (Bearer token)

**Security Note**: Never commit your `.env` file to version control. The `.env.example` file provides a template with placeholder values.

### Customization

- **Vote threshold**: Modify `min_vote_count` in `src/data_loader.py` (default: 1000)
- **Recommendation count**: Change `top_n` parameter in `app.py` (default: 3)
- **Poster size**: Adjust size parameter in `src/tmdb_api.py` (options: w500, w780, original)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Movie data provided by [The Movie Database (TMDB)](https://www.themoviedb.org/)
- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
