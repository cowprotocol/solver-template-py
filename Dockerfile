FROM python:3.11-alpine

# First copy over the requirements.txt and install dependencies, this makes
# building subsequent images easier.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy full source (see .dockerignore)
COPY . .

CMD [ "python3", "-m" , "src._server"]
