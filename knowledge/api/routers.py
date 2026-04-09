import logging
import os
import shutil

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from config.settings import settings
from schemas.schema import QueryRequest, QueryResponse, UploadResponse
from services.ingestion.ingestion_processor import IngestionProcessor
from services.query_service import QueryService
from services.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)

router = APIRouter()

ingestion_processor = IngestionProcessor()
retrieval_service = RetrievalService()
query_service = QueryService()


@router.post("/upload", response_model=UploadResponse, summary="上传知识库文档")
async def upload_file(file: UploadFile = File(...)):
    temp_file_path: str | None = None

    try:
        temp_md_dir = settings.TMP_MD_FOLDER_PATH
        os.makedirs(temp_md_dir, exist_ok=True)

        file_suffix = os.path.splitext(file.filename or "")[1]
        tmp_md_path = os.path.join(temp_md_dir, file.filename or "upload.md")

        async with aiofiles.tempfile.NamedTemporaryFile(
            suffix=file_suffix,
            delete=False,
        ) as temp_file:
            while content := await file.read(1024 * 1024):
                await temp_file.write(content)
            temp_file_path = temp_file.name

        shutil.move(temp_file_path, tmp_md_path)
        chunks_added = await run_in_threadpool(ingestion_processor.ingest_file, tmp_md_path)

        return UploadResponse(
            status="success",
            message="上传知识库成功",
            file_name=file.filename or os.path.basename(tmp_md_path),
            chunks_added=chunks_added,
        )
    except Exception as exc:
        logger.exception("Failed to upload knowledge document.")
        raise HTTPException(status_code=500, detail="文件上传到知识库失败") from exc
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Temporary file removed: %s", temp_file_path)


@router.post("/query", response_model=QueryResponse, summary="查询知识库")
async def query(request: QueryRequest):
    try:
        user_question = (request.question or "").strip()
        if not user_question:
            raise HTTPException(status_code=400, detail="查询问题不能为空")

        query_result = query_service.query(user_question)

        return QueryResponse(
            question=user_question,
            answer=query_result.answer,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Knowledge base query failed.")
        raise HTTPException(status_code=500, detail="服务内部出现异常") from exc
