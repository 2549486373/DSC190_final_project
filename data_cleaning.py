"""
Data Cleaning Script for Survey Data
This script cleans and preprocesses survey data including:
- Column name simplification
- Missing value handling
- Text preprocessing for NLP
- Categorical encoding (one-hot, label encoding, embeddings)
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings



class DataCleaner:
    """
    Comprehensive data cleaning class for survey data
    """

    def __init__(self, filepath='data.csv'):
        """Initialize with data file path"""
        self.filepath = filepath
        self.df = None
        self.df_cleaned = None
        self.column_mapping = {}

    def load_data(self):
        """Load the raw data"""
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self

    def simplify_column_names(self):
        """Simplify long column names to shorter, readable versions"""
        print("\nSimplifying column names...")

        # Define simplified column names
        simplified_names = {
            'Timestamp': 'timestamp',
            'Are you a current UCSD student or staff member?': 'role',
            'Do you live on-campus or off-campus?': 'housing',
            'What do you use as your PRIMARY access for delivery/shipping of packages? (if you use two or more sources equally, please select each one)': 'delivery_method',
            'How many packages/online orders do you have delivered to campus in a typical week?': 'packages_per_week',
            'How many items are usually included in each package/online order you have delivered to campus?': 'items_per_package',
            'How do you dispose of cardboard from packages you have delivered to campus?': 'cardboard_disposal',
            'How do you dispose of plastic (bags, bubble wrap, etc.) from packages you have delivered to campus?': 'plastic_disposal',
            'Please select the item types that you have delivered to campus': 'delivered_items',
            'Which item category/categories in the question above do you have delivered to campus MOST? (you may also specify items purchased/gifted to you)': 'most_delivered',
            'Are there any peak times during the academic year that you order more/less items to be delivered to campus (e.g. move-in, quarter beginning/end, holidays, etc.)? \nIf yes, please describe what time and if the item types you order change, too.': 'peak_times',
            'Please select the options(s) that best describe why you order packages to be delivered to campus instead of shopping in-store:': 'order_reasons',
            'Please select the item categories that you buy in-stores': 'instore_items',
            'Which item category/categories in the question above do you buy in-stores MOST? (you may also specify items purchased)': 'most_instore',
            'Do you typically buy from on or off-campus stores?': 'store_preference',
            'Would you be interested in any of the following shopping options being made more available/frequent? (especially to reduce online ordering)': 'shopping_interests',
            'Are there any items/goods that you would like the on-campus stores/vendors to offer? \n(identifying specific items, times they should be offered, and preferred price range is encouraged)': 'item_requests',
            'What communication method(s) do you get most of your campus news from?': 'communication_methods',
            'Is there anything else you would like us to know?': 'additional_comments'
        }

        self.df_cleaned = self.df.rename(columns=simplified_names)
        self.column_mapping = simplified_names

        print(f"Renamed {len(simplified_names)} columns")
        print("New columns:", list(self.df_cleaned.columns))
        return self

    def handle_missing_values(self, strategy='smart'):
        """
        Handle missing values with different strategies

        Args:
            strategy: 'smart' (default), 'drop', 'fill_unknown', or 'fill_mode'
        """
        print(f"\nHandling missing values with strategy: {strategy}")

        # Show missing value counts
        missing_counts = self.df_cleaned.isnull().sum()
        print("\nMissing values per column:")
        print(missing_counts[missing_counts > 0])

        if strategy == 'smart':
            # Smart strategy: different handling for different column types

            # For numerical-like columns, fill with 0 or mode
            numeric_like = ['packages_per_week', 'items_per_package']
            for col in numeric_like:
                if col in self.df_cleaned.columns:
                    self.df_cleaned[col].fillna('0', inplace=True)

            # For categorical columns, fill with 'Not Answered'
            categorical = ['delivery_method', 'cardboard_disposal', 'plastic_disposal',
                          'delivered_items', 'order_reasons', 'shopping_interests']
            for col in categorical:
                if col in self.df_cleaned.columns:
                    self.df_cleaned[col].fillna('Not Answered', inplace=True)

            # For text columns, fill with empty string
            text_cols = ['most_delivered', 'peak_times', 'most_instore',
                        'item_requests', 'additional_comments']
            for col in text_cols:
                if col in self.df_cleaned.columns:
                    self.df_cleaned[col].fillna('', inplace=True)

        elif strategy == 'drop':
            # Drop rows with any missing values
            before = len(self.df_cleaned)
            self.df_cleaned = self.df_cleaned.dropna()
            print(f"Dropped {before - len(self.df_cleaned)} rows")

        elif strategy == 'fill_unknown':
            # Fill all missing with 'Unknown'
            self.df_cleaned = self.df_cleaned.fillna('Unknown')

        elif strategy == 'fill_mode':
            # Fill with mode (most common value) for each column
            for col in self.df_cleaned.columns:
                if self.df_cleaned[col].isnull().any():
                    mode_val = self.df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        self.df_cleaned[col].fillna(mode_val[0], inplace=True)

        print(f"\nRemaining missing values: {self.df_cleaned.isnull().sum().sum()}")
        return self

    def preprocess_text_simple(self, text):
        """
        Simple text preprocessing for NLP
        - Lowercase
        - Remove special characters
        - Strip whitespace
        """
        if pd.isna(text) or text == '':
            return ''

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def clean_text_columns(self, columns=None):
        """
        Clean text columns with basic preprocessing

        Args:
            columns: List of column names to clean. If None, cleans all text columns
        """
        print("\nCleaning text columns...")

        if columns is None:
            # Default text columns
            columns = ['most_delivered', 'peak_times', 'most_instore',
                      'item_requests', 'additional_comments']

        for col in columns:
            if col in self.df_cleaned.columns:
                self.df_cleaned[f'{col}_cleaned'] = self.df_cleaned[col].apply(
                    self.preprocess_text_simple
                )
                print(f"Cleaned column: {col}")

        return self

    def extract_multi_select_categories(self, column):
        """
        Extract unique categories from multi-select columns (comma-separated)

        Args:
            column: Column name with comma-separated values

        Returns:
            List of unique categories
        """
        all_categories = set()

        for value in self.df_cleaned[column].dropna():
            if value and value != 'Not Answered':
                # Split by comma and clean
                categories = [cat.strip() for cat in str(value).split(',')]
                all_categories.update(categories)

        return sorted(list(all_categories))

    def one_hot_encode_multi_select(self, columns=None):
        """
        One-hot encode multi-select categorical columns

        Args:
            columns: List of column names with comma-separated values
        """
        print("\nOne-hot encoding multi-select columns...")

        if columns is None:
            columns = ['delivered_items', 'order_reasons', 'instore_items',
                      'communication_methods', 'shopping_interests']

        for col in columns:
            if col not in self.df_cleaned.columns:
                continue

            print(f"\nProcessing: {col}")
            categories = self.extract_multi_select_categories(col)
            print(f"Found {len(categories)} unique categories")

            # Create binary columns for each category
            for category in categories:
                col_name = f"{col}_{category.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').lower()}"
                self.df_cleaned[col_name] = self.df_cleaned[col].apply(
                    lambda x: 1 if category in str(x) else 0
                )

        return self

    def label_encode_single_select(self, columns=None):
        """
        Label encode single-select categorical columns

        Args:
            columns: List of column names to label encode
        """
        print("\nLabel encoding single-select columns...")

        if columns is None:
            columns = ['role', 'housing', 'store_preference', 'packages_per_week']

        encoders = {}

        for col in columns:
            if col not in self.df_cleaned.columns:
                continue

            le = LabelEncoder()
            # Handle any remaining NaN values
            valid_mask = self.df_cleaned[col].notna()
            self.df_cleaned.loc[valid_mask, f'{col}_encoded'] = le.fit_transform(
                self.df_cleaned.loc[valid_mask, col]
            )
            encoders[col] = le

            print(f"Encoded {col}: {len(le.classes_)} classes")
            print(f"  Classes: {list(le.classes_)}")

        return encoders

    def create_tfidf_features(self, text_columns=None, max_features=50):
        """
        Create TF-IDF features from text columns

        Args:
            text_columns: List of text column names
            max_features: Maximum number of features per column

        Returns:
            Dictionary of fitted TF-IDF vectorizers
        """
        print(f"\nCreating TF-IDF features (max {max_features} features per column)...")

        if text_columns is None:
            text_columns = ['most_delivered_cleaned', 'most_instore_cleaned',
                           'peak_times_cleaned']

        vectorizers = {}

        for col in text_columns:
            if col not in self.df_cleaned.columns:
                continue

            # Skip if all empty
            if self.df_cleaned[col].str.len().sum() == 0:
                print(f"Skipping {col} - all empty")
                continue

            tfidf = TfidfVectorizer(max_features=max_features, min_df=2,
                                   stop_words='english')

            try:
                tfidf_matrix = tfidf.fit_transform(self.df_cleaned[col])

                # Create columns for each feature
                feature_names = tfidf.get_feature_names_out()
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'{col}_tfidf_{name}' for name in feature_names]
                )

                # Add to main dataframe
                self.df_cleaned = pd.concat([self.df_cleaned, tfidf_df], axis=1)
                vectorizers[col] = tfidf

                print(f"Created {len(feature_names)} TF-IDF features for {col}")

            except Exception as e:
                print(f"Error processing {col}: {e}")

        return vectorizers

    def create_openai_embeddings(self, text_columns=None, api_key=None, model='text-embedding-3-small'):
        """
        Create embeddings using OpenAI API

        Args:
            text_columns: List of text column names
            api_key: OpenAI API key (if not set as environment variable)
            model: OpenAI embedding model to use

        Returns:
            Dictionary mapping column names to embedding DataFrames
        """
        if not OPENAI_AVAILABLE:
            print("OpenAI package not available. Please install: pip install openai")
            return {}

        if api_key:
            openai.api_key = api_key

        print(f"\nCreating OpenAI embeddings using model: {model}...")
        print("WARNING: This will make API calls and may incur costs!")

        if text_columns is None:
            text_columns = ['most_delivered_cleaned', 'peak_times_cleaned',
                           'item_requests_cleaned']

        embeddings_dict = {}

        for col in text_columns:
            if col not in self.df_cleaned.columns:
                continue

            print(f"\nProcessing {col}...")

            # Filter out empty texts
            texts = self.df_cleaned[col].fillna('').tolist()
            embeddings = []

            for i, text in enumerate(texts):
                if text.strip() == '':
                    # Use zero vector for empty texts
                    embeddings.append([0.0] * 1536)  # Default embedding size
                else:
                    try:
                        response = openai.embeddings.create(
                            input=text,
                            model=model
                        )
                        embeddings.append(response.data[0].embedding)

                        if (i + 1) % 10 == 0:
                            print(f"  Processed {i + 1}/{len(texts)} texts")

                    except Exception as e:
                        print(f"  Error embedding text {i}: {e}")
                        embeddings.append([0.0] * 1536)

            # Create DataFrame with embeddings
            embedding_dim = len(embeddings[0])
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'{col}_embed_{i}' for i in range(embedding_dim)]
            )

            # Add to main dataframe
            self.df_cleaned = pd.concat([self.df_cleaned, embedding_df], axis=1)
            embeddings_dict[col] = embedding_df

            print(f"Created {embedding_dim} embedding features for {col}")

        return embeddings_dict

    def get_feature_columns_for_ml(self):
        """
        Get list of feature columns suitable for ML models (numeric only)

        Returns:
            List of column names
        """
        # Get all numeric columns and encoded columns
        feature_cols = []

        for col in self.df_cleaned.columns:
            # Include encoded columns, one-hot columns, tfidf, embeddings
            if any(suffix in col for suffix in ['_encoded', '_tfidf_', '_embed_']) or \
               self.df_cleaned[col].dtype in ['int64', 'float64']:
                feature_cols.append(col)

        return feature_cols

    def save_cleaned_data(self, output_path='data_cleaned.csv'):
        """Save cleaned data to CSV"""
        print(f"\nSaving cleaned data to {output_path}...")
        self.df_cleaned.to_csv(output_path, index=False)
        print(f"Saved {len(self.df_cleaned)} rows and {len(self.df_cleaned.columns)} columns")
        return self

    def get_summary(self):
        """Print summary of cleaned data"""
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Original shape: {self.df.shape}")
        print(f"Cleaned shape: {self.df_cleaned.shape}")
        print(f"\nColumns added: {self.df_cleaned.shape[1] - self.df.shape[1]}")
        print(f"Missing values: {self.df_cleaned.isnull().sum().sum()}")
        print(f"\nData types:")
        print(self.df_cleaned.dtypes.value_counts())

        # Get ML-ready features
        ml_features = self.get_feature_columns_for_ml()
        print(f"\nML-ready feature columns: {len(ml_features)}")

        return self


# ============================================================================
# MAIN CLEANING PIPELINE
# ============================================================================

def clean_data_pipeline(
    input_file='data.csv',
    output_file='data_cleaned.csv',
    use_tfidf=True,
    use_openai=False,
    openai_api_key=None
):
    """
    Main data cleaning pipeline with multiple options

    Args:
        input_file: Path to raw data CSV
        output_file: Path to save cleaned data
        use_tfidf: Whether to create TF-IDF features (recommended)
        use_openai: Whether to create OpenAI embeddings (requires API key)
        openai_api_key: OpenAI API key (optional if set as env variable)
    """

    # Initialize cleaner
    cleaner = DataCleaner(input_file)

    # Step 1: Load and simplify
    cleaner.load_data()
    cleaner.simplify_column_names()

    # Step 2: Handle missing values
    cleaner.handle_missing_values(strategy='smart')

    # Step 3: Clean text columns
    cleaner.clean_text_columns()

    # Step 4: Encode categorical variables
    # One-hot encode multi-select columns
    cleaner.one_hot_encode_multi_select()

    # Label encode single-select columns
    label_encoders = cleaner.label_encode_single_select()

    # Step 5: Create text features
    if use_tfidf:
        tfidf_vectorizers = cleaner.create_tfidf_features(max_features=30)

    if use_openai and openai_api_key:
        print("\nNote: OpenAI embeddings will be created (this costs money!)")
        response = input("Continue? (yes/no): ")
        if response.lower() == 'yes':
            embeddings = cleaner.create_openai_embeddings(api_key=openai_api_key)

    # Step 6: Save and summarize
    cleaner.save_cleaned_data(output_file)
    cleaner.get_summary()

    # Print example ML-ready columns
    ml_cols = cleaner.get_feature_columns_for_ml()
    print(f"\nExample ML-ready columns (first 10):")
    for col in ml_cols[:10]:
        print(f"  - {col}")

    return cleaner


if __name__ == "__main__":
    """
    Example usage
    """

    print("="*60)
    print("SURVEY DATA CLEANING SCRIPT")
    print("="*60)

    # Option 1: Basic cleaning with TF-IDF (recommended)
    print("\n[Option 1] Running basic cleaning with TF-IDF...")
    cleaner = clean_data_pipeline(
        input_file='data.csv',
        output_file='data_cleaned.csv',
        use_tfidf=True,
        use_openai=False
    )

    # Option 2: With OpenAI embeddings (uncomment to use)
    # print("\n[Option 2] Running cleaning with OpenAI embeddings...")
    # cleaner = clean_data_pipeline(
    #     input_file='data.csv',
    #     output_file='data_cleaned_embeddings.csv',
    #     use_tfidf=True,
    #     use_openai=True,
    #     openai_api_key='your-api-key-here'  # Or set OPENAI_API_KEY env variable
    # )

    print("\n" + "="*60)
    print("CLEANING COMPLETE!")
    print("="*60)
    print(f"\nCleaned data saved to: data_cleaned.csv")
    print("\nNext steps:")
    print("1. Load cleaned data: df = pd.read_csv('data_cleaned.csv')")
    print("2. Get ML features: features = [col for col in df.columns if '_encoded' in col or '_tfidf_' in col]")
    print("3. Train model: RandomForestClassifier, etc.")
