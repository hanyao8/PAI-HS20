FROM python:3.8.5-slim
ADD ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN apt-get update
RUN apt-get install -y xvfb python-opengl ffmpeg
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
WORKDIR /code
CMD xvfb-run -s "-screen 0 1400x900x24" python -u checker_client.py
