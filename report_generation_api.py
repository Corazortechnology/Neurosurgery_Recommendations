from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from report_generation import ReportGenerator
import os

app = FastAPI()
generator = ReportGenerator()

class ReportRequest(BaseModel):
    start_date: str
    end_date: str 
    user_id: str

@app.post("/generate-report")
def generate_report(request: ReportRequest, background_tasks: BackgroundTasks):
    reports = generator.generate_reports_for_period(request.start_date, request.end_date, request.user_id)
    if not reports or request.user_id not in reports:
        raise HTTPException(status_code=404, detail=f"No report found for user {request.user_id} in the given period.")

    file_path = reports[request.user_id]
    background_tasks.add_task(os.remove, file_path)
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/pdf"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)