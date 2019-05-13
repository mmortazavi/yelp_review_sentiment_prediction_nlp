#Base image
FROM  python:3.6 
#gw000/keras:2.1.4-py3-tf-cpu 
#python:3.6

# Updating repository sources
RUN apt-get update
# RUN apt-get update -qq \
#  && apt-get install --no-install-recommends -y \
#     python-matplotlib \

# Copying requirements.txt file
COPY ./requirements.txt /requirements.txt

# pip install all requirements
RUN pip install --no-cache -r requirements.txt
RUN python -m nltk.downloader stopwords

# Setting up working directory
WORKDIR /

# Copy from local host to Docker image
COPY . /

# Training the Model
CMD ["python", "setup.py"]
