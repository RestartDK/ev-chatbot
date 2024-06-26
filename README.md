# EV Barbie Chatbot

Welcome to the EV Barbie Chatbot project! This interactive chatbot is designed to assist you in exploring and purchasing electric vehicles (EV) by responding dynamically to your emotions. Using advanced language models from LangChain and the intuitive interface capabilities of Streamlit, the EV Barbie Chatbot offers a user-friendly and engaging shopping experience.

## Features

- **Emotion Detection**: The chatbot analyzes the user's emotional tone to tailor responses and provide a personalized experience.
- **EV Information**: Provides detailed information on various electric vehicles, including specs, pricing, and environmental benefits.
- **Interactive UI**: Built with Streamlit, the interface is clean and simple, making it easy for users of all ages to navigate.
- **Persona Mimicry**: The chatbot mimics the friendly and supportive personality of Barbie to make the car buying process fun and engaging.

## How It Works

The chatbot uses a combination of LangChain for processing natural language and understanding user emotions, and Streamlit for creating a responsive web interface. Here’s a brief on how these components work together:
- **LangChain**: Handles all backend language understanding tasks, emotion detection, and response generation.
- **Streamlit**: Provides the frontend interface that interacts with the user, displaying responses and collecting user inputs.

## Getting Started

Follow these instructions to set up and run the EV Barbie Chatbot on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python (3.7 or later)
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ev-barbie-chatbot.git
   cd barbie-chat
   ```

2. **Install required Python packages**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit server**

   ```bash
   streamlit run main.py
   ```

2. **Open your web browser**

   Navigate to `http://localhost:8501` to view and interact with the chatbot.

## Usage

- Simply type your questions or statements into the chat interface. The chatbot will process your emotions and the content of your message to generate a response that fits your emotional state and query.
- Examples of questions you might ask:
  - "Tell me more about the Tesla Model S performance."
  - "I’m not sure which EV fits my budget."
  - "Can you explain the benefits of switching to an electric vehicle?"