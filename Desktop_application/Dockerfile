FROM ubuntu:18.04

RUN apt-get update

# Install npm nodeJS
RUN apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs

# Install libraries
RUN apt-get -y install libgtkextra-dev libgconf2-dev libnss3 libasound2 libxtst-dev libxss1 libx11-xcb1
RUN apt-get install -y libglib2.0-0 \
                       libnss3 \
                       libgdk-pixbuf2.0-0 \
                       libgtk-3-0 

# Copy source code
COPY . /app

# Change working directory
WORKDIR /app

#Install electron
RUN npm install --save-dev electron


# Launch application
CMD ["npm","start"]
