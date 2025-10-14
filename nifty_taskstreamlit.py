{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f38d2a10-c8e0-46ac-aaea-b31e339b8ca7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-10-13 19:50:21.594 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\soumy\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Streamlit title\n",
    "st.title(\"📈 Nifty Stock Data Analysis\")\n",
    "\n",
    "# File uploader\n",
    "uploaded_file = st.file_uploader(\"Upload your CSV file\", type=[\"csv\", \"xlsx\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # Read the file\n",
    "    if uploaded_file.name.endswith(\".csv\"):\n",
    "        df = pd.read_csv(uploaded_file)\n",
    "    else:\n",
    "        df = pd.read_excel(uploaded_file)\n",
    "\n",
    "    st.subheader(\"Preview of Uploaded Data\")\n",
    "    st.dataframe(df.head())\n",
    "\n",
    "    # Drop columns 'Close' and 'Date'\n",
    "    if 'Close' in df.columns and 'Date' in df.columns:\n",
    "        x = df.drop(['Close', 'Date'], axis=1).values\n",
    "        y = df['Close'].values\n",
    "\n",
    "        st.success(\"✅ Columns dropped successfully!\")\n",
    "        st.write(f\"Shape of X: {x.shape}\")\n",
    "        st.write(f\"Shape of y: {y.shape}\")\n",
    "\n",
    "        # Optionally visualize\n",
    "        st.subheader(\"Close Price Over Time\")\n",
    "        st.line_chart(df.set_index('Date')['Close'])\n",
    "    else:\n",
    "        st.error(\"⚠️ The dataset must contain 'Close' and 'Date' columns.\")\n",
    "else:\n",
    "    st.info(\"Please upload a dataset to begin.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ee143e2b-129e-450b-95ee-c2237afcac91",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90f1cf76-0a4e-48be-8d94-234e1d81b011",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

