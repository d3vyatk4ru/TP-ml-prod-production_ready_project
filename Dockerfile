FROM python:3.8

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY sdist/Production_ready_project_hw_1.egg-info /Production_ready_project_hw_1.egg-info
RUN pip install /Production_ready_project_hw_1.egg-info

COPY configs/ /configs

WORKDIR .

CMD ["python", "src/train_pipeline.py"]