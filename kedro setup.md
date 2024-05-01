

# create kedro project
```ps1
conda activate asi
kedro new --name pjatk_asi --tools=docs,data --example=y
cd .\pjatk_asi
pip install -r requirements.txt
kedro run
kedro viz run
```



