from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from .. import crud, schemas, utils


router = APIRouter()


@router.post("/score", response_model=schemas.MatchOut)
def score_match(payload: schemas.MatchRequest, db: Session = Depends(get_db)):
	resume = crud.get_resume(db, payload.resume_id)
	job = crud.get_job(db, payload.job_id)
	if not resume or not job:
		raise HTTPException(status_code=404, detail="Resume or Job not found")

	if not resume.embedding:
		resume_emb = utils.encode(resume.text)
	else:
		resume_emb = resume.embedding

	if not job.embedding:
		job_emb = utils.encode(job.description)
	else:
		job_emb = job.embedding

	score = utils.cosine_similarity(resume_emb, job_emb)
	job_keywords = utils.extract_top_keywords(job.description)
	missing = utils.diff_keywords(job_keywords, resume.text)
	suggestions = utils.formatting_suggestions(resume.text)

	match = crud.upsert_match(
		db,
		resume_id=resume.id,
		job_id=job.id,
		score=score,
		missing_keywords=missing,
		suggestions=suggestions,
	)
	return match


@router.post("/rank_candidates", response_model=schemas.RankCandidatesResponse)
def rank_candidates(payload: schemas.RankCandidatesRequest, db: Session = Depends(get_db)):
	job = crud.get_job(db, payload.job_id)
	if not job:
		raise HTTPException(status_code=404, detail="Job not found")
	job_emb = job.embedding or utils.encode(job.description)

	results: List[schemas.RankedCandidate] = []
	for resume_id in payload.resume_ids:
		resume = crud.get_resume(db, resume_id)
		if not resume:
			continue
		resume_emb = resume.embedding or utils.encode(resume.text)
		score = utils.cosine_similarity(resume_emb, job_emb)
		job_keywords = utils.extract_top_keywords(job.description)
		missing = utils.diff_keywords(job_keywords, resume.text)
		suggestions = utils.formatting_suggestions(resume.text)
		crud.upsert_match(db, resume_id=resume.id, job_id=job.id, score=score, missing_keywords=missing, suggestions=suggestions)
		results.append(
			schemas.RankedCandidate(
				resume_id=resume.id,
				score=score,
				missing_keywords=missing,
				suggestions=suggestions,
			)
		)

	results.sort(key=lambda r: r.score, reverse=True)
	return schemas.RankCandidatesResponse(job_id=job.id, results=results[: payload.limit])


