FROM python:alpine
RUN apk add --update gcc libc-dev linux-headers

WORKDIR /app

# Only copies requirements.txt and src directory (see .dockerignore)
COPY . .
RUN pip install -r requirements.txt

ENTRYPOINT [ "python3", "-m" , "src._server"]