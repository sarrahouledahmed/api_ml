FROM python:3.8.0
# define the present working directory
ADD . /code
WORKDIR /code
RUN pip install forex_python
# run pip to install the dependencies of the flask app
COPY requirements.txt requirements.txt
COPY ./app.py ./app.py
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY .. .
# define the command to start the container
CMD [ "python3", "app.py"]
