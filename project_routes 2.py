"""
YMERA Project Management Routes
Project creation, management, and tracking endpoints
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.connection import get_db, Project
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Pydantic models
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    repository_url: Optional[str] = None
    owner_id: str  # In production, this would come from authentication

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    repository_url: Optional[str] = None
    status: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    repository_url: Optional[str]
    status: str
    owner_id: str
    created_at: datetime
    updated_at: datetime

class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new project"""
    
    # Check if project name already exists for this owner
    result = await db.execute(
        select(Project).where(
            (Project.name == project_data.name) & 
            (Project.owner_id == project_data.owner_id)
        )
    )
    existing_project = result.scalar_one_or_none()
    
    if existing_project:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project with this name already exists"
        )
    
    # Create new project
    new_project = Project(
        name=project_data.name,
        description=project_data.description,
        repository_url=project_data.repository_url,
        owner_id=project_data.owner_id,
        status="active"
    )
    
    db.add(new_project)
    await db.commit()
    await db.refresh(new_project)
    
    logger.info(f"New project created: {project_data.name} by {project_data.owner_id}")
    
    return ProjectResponse(
        id=new_project.id,
        name=new_project.name,
        description=new_project.description,
        repository_url=new_project.repository_url,
        status=new_project.status,
        owner_id=new_project.owner_id,
        created_at=new_project.created_at,
        updated_at=new_project.updated_at
    )

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get project by ID"""
    
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        repository_url=project.repository_url,
        status=project.status,
        owner_id=project.owner_id,
        created_at=project.created_at,
        updated_at=project.updated_at
    )

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update project"""
    
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Update fields if provided
    if project_data.name is not None:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description
    if project_data.repository_url is not None:
        project.repository_url = project_data.repository_url
    if project_data.status is not None:
        project.status = project_data.status
    
    project.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(project)
    
    logger.info(f"Project updated: {project.name}")
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        repository_url=project.repository_url,
        status=project.status,
        owner_id=project.owner_id,
        created_at=project.created_at,
        updated_at=project.updated_at
    )

@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    owner_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List projects with optional filtering"""
    
    query = select(Project)
    
    # Apply filters
    if owner_id:
        query = query.where(Project.owner_id == owner_id)
    if status:
        query = query.where(Project.status == status)
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    projects = result.scalars().all()
    
    # Get total count (simplified for Phase 1-2)
    total = len(projects)
    
    project_responses = [
        ProjectResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            repository_url=p.repository_url,
            status=p.status,
            owner_id=p.owner_id,
            created_at=p.created_at,
            updated_at=p.updated_at
        )
        for p in projects
    ]
    
    return ProjectListResponse(projects=project_responses, total=total)

@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete project"""
    
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # In a production system, you'd also handle cascade deletion of related files, etc.
    
    await db.delete(project)
    await db.commit()
    
    logger.info(f"Project deleted: {project.name}")
    
    return {"message": "Project deleted successfully"}

@router.get("/{project_id}/files")
async def get_project_files(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get files associated with a project"""
    
    # First verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Get files for this project
    from database.connection import File
    result = await db.execute(select(File).where(File.project_id == project_id))
    files = result.scalars().all()
    
    return {
        "project_id": project_id,
        "project_name": project.name,
        "files": [
            {
                "id": f.id,
                "filename": f.original_filename,
                "size": f.file_size,
                "content_type": f.content_type,
                "created_at": f.created_at
            }
            for f in files
        ],
        "total_files": len(files)
    }

@router.get("/status/system")
async def project_system_status():
    """Project management system status"""
    return {
        "status": "operational",
        "features": {
            "create_project": "active",
            "update_project": "active",
            "delete_project": "active",
            "list_projects": "active",
            "project_files": "active",
            "project_permissions": "phase_3",
            "project_collaboration": "phase_3"
        },
        "phase": "1-2"
    }