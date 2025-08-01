#!/bin/bash

# ===============================================================================
# YMERA Enterprise Platform - Startup Script
# Production-Ready Initialization & Service Management - v4.0
# Enterprise-grade implementation with zero placeholders
# ===============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ===============================================================================
# SCRIPT CONFIGURATION
# ===============================================================================

# Script metadata
readonly SCRIPT_NAME="ymera_startup_script.sh"
readonly SCRIPT_VERSION="4.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Logging configuration
readonly LOG_DIR="/tmp/ymera_logs"
readonly LOG_FILE="$LOG_DIR/startup.log"
readonly ERROR_LOG="$LOG_DIR/startup_errors.log"

# Service configuration
readonly SERVICE_NAME="ymera-platform"
readonly PYTHON_VERSION="3.11"
readonly MIN_PYTHON_VERSION="3.9"

# Environment detection
readonly ENVIRONMENT="${ENVIRONMENT:-development}"
readonly IS_REPLIT="${REPLIT:-false}"
readonly IS_DOCKER="${DOCKER:-false}"

# Performance settings
readonly MAX_WORKERS="${WORKERS:-4}"
readonly TIMEOUT="${TIMEOUT:-30}"
readonly KEEP_ALIVE="${KEEP_ALIVE:-2}"

# ===============================================================================
# LOGGING FUNCTIONS
# ===============================================================================

setup_logging() {
    mkdir -p "$LOG_DIR"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$ERROR_LOG" >&2)
}

log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
}

log_warn() {
    log "WARN" "$@"
}

log_error() {
    log "ERROR" "$@"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >> "$ERROR_LOG"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        log "DEBUG" "$@"
    fi
}

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command '$1' not found"
        return 1
    fi
    return 0
}

check_python_version() {
    local python_cmd="$1"
    local version
    version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
    local major_minor
    major_minor=$(echo "$version" | cut -d'.' -f1,2)
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        log_info "Python version $version is compatible"
        return 0
    else
        log_error "Python version $version is not compatible (minimum: $MIN_PYTHON_VERSION)"
        return 1
    fi
}

wait_for_service() {
    local service_name="$1"
    local host="${2:-localhost}"
    local port="$3"
    local timeout="${4:-30}"
    local interval="${5:-2}"
    
    log_info "Waiting for $service_name to be ready on $host:$port..."
    
    local elapsed=0
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [[ $elapsed -ge $timeout ]]; then
            log_error "$service_name failed to start within $timeout seconds"
            return 1
        fi
        
        sleep "$interval"
        elapsed=$((elapsed + interval))
        log_debug "Waiting for $service_name... (${elapsed}s elapsed)"
    done
    
    log_info "$service_name is ready!"
    return 0
}

cleanup_on_exit() {
    local exit_code=$?
    log_info "Cleanup initiated (exit code: $exit_code)"
    
    # Kill background processes
    if [[ -n "${YMERA_PID:-}" ]]; then
        if kill -0 "$YMERA_PID" 2>/dev/null; then
            log_info "Stopping YMERA main process (PID: $YMERA_PID)"
            kill -TERM "$YMERA_PID" 2>/dev/null || true
            sleep 5
            if kill -0 "$YMERA_PID" 2>/dev/null; then
                kill -KILL "$YMERA_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    # Stop background services
    stop_background_services
    
    log_info "Cleanup completed"
    exit $exit_code
}

# ===============================================================================
# ENVIRONMENT SETUP FUNCTIONS
# ===============================================================================

detect_environment() {
    log_info "Detecting runtime environment..."
    
    if [[ "$IS_REPLIT" == "true" || -n "${REPL_SLUG:-}" ]]; then
        export RUNTIME_ENVIRONMENT="replit"
        export DATABASE_URL="${REPLIT_DB_URL:-postgresql://postgres:password@localhost:5432/ymera}"
        export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
        log_info "Environment: Replit detected"
    elif [[ "$IS_DOCKER" == "true" || -f "/.dockerenv" ]]; then
        export RUNTIME_ENVIRONMENT="docker"
        export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:password@db:5432/ymera}"
        export REDIS_URL="${REDIS_URL:-redis://redis:6379}"
        log_info "Environment: Docker detected"
    elif [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
        export RUNTIME_ENVIRONMENT="kubernetes"
        log_info "Environment: Kubernetes detected"
    else
        export RUNTIME_ENVIRONMENT="local"
        export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:password@localhost:5432/ymera}"
        export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
        log_info "Environment: Local development detected"
    fi
    
    # Set environment-specific configurations
    case "$RUNTIME_ENVIRONMENT" in
        "replit")
            export HOST="0.0.0.0"
            export PORT="${PORT:-8000}"
            export WORKERS="2"
            ;;
        "docker")
            export HOST="0.0.0.0"
            export PORT="${PORT:-8000}"
            export WORKERS="${MAX_WORKERS}"
            ;;
        "kubernetes")
            export HOST="0.0.0.0"
            export PORT="${PORT:-8000}"
            export WORKERS="${MAX_WORKERS}"
            ;;
        "local")
            export HOST="${HOST:-127.0.0.1}"
            export PORT="${PORT:-8000}"
            export WORKERS="1"
            ;;
    esac
    
    log_info "Runtime configuration: HOST=$HOST, PORT=$PORT, WORKERS=$WORKERS"
}

setup_python_environment() {
    log_info "Setting up Python environment..."
    
    # Find Python executable
    local python_cmd=""
    for cmd in python3.11 python3.10 python3.9 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            if check_python_version "$cmd"; then
                python_cmd="$cmd"
                break
            fi
        fi
    done
    
    if [[ -z "$python_cmd" ]]; then
        log_error "No compatible Python version found"
        return 1
    fi
    
    export PYTHON_CMD="$python_cmd"
    log_info "Using Python: $python_cmd"
    
    # Set Python path
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
    
    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_warn "No virtual environment detected"
        
        # Try to activate virtual environment if it exists
        if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
            log_info "Activating virtual environment..."
            source "${PROJECT_ROOT}/venv/bin/activate"
        elif [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
            log_info "Activating virtual environment..."
            source "${PROJECT_ROOT}/.venv/bin/activate"
        else
            log_warn "No virtual environment found, using system Python"
        fi
    else
        log_info "Virtual environment active: $VIRTUAL_ENV"
    fi
    
    return 0
}

install_dependencies() {
    log_info "Installing/updating Python dependencies..."
    
    # Upgrade pip first
    if [[ "$RUNTIME_ENVIRONMENT" != "replit" ]]; then
        log_info "Upgrading pip..."
        $PYTHON_CMD -m pip install --upgrade pip || {
            log_warn "Failed to upgrade pip, continuing..."
        }
    fi
    
    # Install requirements
    if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        log_info "Installing requirements from requirements.txt..."
        $PYTHON_CMD -m pip install -r "${PROJECT_ROOT}/requirements.txt" || {
            log_error "Failed to install requirements"
            return 1
        }
    elif [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        log_info "Installing dependencies from pyproject.toml..."
        $PYTHON_CMD -m pip install -e "${PROJECT_ROOT}" || {
            log_error "Failed to install project dependencies"
            return 1
        }
    else
        log_warn "No requirements.txt or pyproject.toml found"
    fi
    
    # Verify critical dependencies
    local critical_deps=("fastapi" "uvicorn" "sqlalchemy" "redis" "structlog")
    for dep in "${critical_deps[@]}"; do
        if ! $PYTHON_CMD -c "import $dep" &>/dev/null; then
            log_warn "Critical dependency '$dep' not found, attempting to install..."
            $PYTHON_CMD -m pip install "$dep" || {
                log_error "Failed to install critical dependency: $dep"
                return 1
            }
        fi
    done
    
    log_info "Dependencies installation completed"
    return 0
}

# ===============================================================================
# SERVICE MANAGEMENT FUNCTIONS
# ===============================================================================

start_database_services() {
    log_info "Starting database services..."
    
    case "$RUNTIME_ENVIRONMENT" in
        "replit")
            # Replit handles PostgreSQL automatically
            log_info "PostgreSQL managed by Replit"
            ;;
        "docker")
            # Docker Compose handles services
            log_info "Database services managed by Docker"
            ;;
        "local")
            # Start local PostgreSQL if needed
            if ! pgrep -f "postgres" > /dev/null; then
                log_info "Starting local PostgreSQL..."
                if command -v brew &> /dev/null && brew services list | grep -q postgresql; then
                    brew services start postgresql || log_warn "Failed to start PostgreSQL via brew"
                elif command -v systemctl &> /dev/null; then
                    sudo systemctl start postgresql || log_warn "Failed to start PostgreSQL via systemctl"
                elif command -v service &> /dev/null; then
                    sudo service postgresql start || log_warn "Failed to start PostgreSQL via service"
                else
                    log_warn "Could not start PostgreSQL automatically"
                fi
            fi
            
            # Start Redis if needed
            if ! pgrep -f "redis-server" > /dev/null; then
                log_info "Starting local Redis..."
                if command -v redis-server &> /dev/null; then
                    redis-server --daemonize yes --logfile "$LOG_DIR/redis.log" || {
                        log_warn "Failed to start Redis as daemon, trying foreground..."
                        nohup redis-server > "$LOG_DIR/redis.log" 2>&1 &
                    }
                else
                    log_warn "Redis not found, some features may not work"
                fi
            fi
            ;;
    esac
    
    # Wait for services to be ready
    if [[ "$RUNTIME_ENVIRONMENT" != "docker" ]]; then
        wait_for_service "PostgreSQL" "localhost" "5432" 30 || {
            log_error "PostgreSQL is not ready"
            return 1
        }
        
        wait_for_service "Redis" "localhost" "6379" 30 || {
            log_warn "Redis is not ready, continuing without cache"
        }
    fi
    
    return 0
}

setup_database() {
    log_info "Setting up database..."
    
    # Run database initialization script
    local db_init_script="${PROJECT_ROOT}/database/init_db.py"
    if [[ -f "$db_init_script" ]]; then
        log_info "Running database initialization..."
        $PYTHON_CMD "$db_init_script" || {
            log_error "Database initialization failed"
            return 1
        }
    fi
    
    # Run migrations
    local alembic_dir="${PROJECT_ROOT}/alembic"
    if [[ -d "$alembic_dir" && -f "$alembic_dir/alembic.ini" ]]; then
        log_info "Running database migrations..."
        cd "$PROJECT_ROOT"
        $PYTHON_CMD -m alembic upgrade head || {
            log_error "Database migration failed"
            return 1
        }
    else
        log_warn "No Alembic configuration found, skipping migrations"
    fi
    
    # Create initial data
    local seed_script="${PROJECT_ROOT}/scripts/seed_data.py"
    if [[ -f "$seed_script" ]]; then
        log_info "Seeding initial data..."
        $PYTHON_CMD "$seed_script" || {
            log_warn "Data seeding failed, continuing..."
        }
    fi
    
    return 0
}

start_background_services() {
    log_info "Starting background services..."
    
    # Start Redis if not already running
    if ! pgrep -f "redis-server" > /dev/null && [[ "$RUNTIME_ENVIRONMENT" == "local" ]]; then
        log_info "Starting Redis server..."
        nohup redis-server > "$LOG_DIR/redis.log" 2>&1 &
        local redis_pid=$!
        echo "$redis_pid" > "$LOG_DIR/redis.pid"
    fi
    
    # Start task queue worker
    local worker_script="${PROJECT_ROOT}/workers/task_worker.py"
    if [[ -f "$worker_script" ]]; then
        log_info "Starting task queue worker..."
        nohup $PYTHON_CMD "$worker_script" > "$LOG_DIR/worker.log" 2>&1 &
        local worker_pid=$!
        echo "$worker_pid" > "$LOG_DIR/worker.pid"
        log_info "Task worker started (PID: $worker_pid)"
    fi
    
    # Start agent orchestrator
    local orchestrator_script="${PROJECT_ROOT}/ymera_agents/orchestration_agent.py"
    if [[ -f "$orchestrator_script" ]]; then
        log_info "Starting agent orchestrator..."
        nohup $PYTHON_CMD "$orchestrator_script" > "$LOG_DIR/orchestrator.log" 2>&1 &
        local orchestrator_pid=$!
        echo "$orchestrator_pid" > "$LOG_DIR/orchestrator.pid"
        log_info "Agent orchestrator started (PID: $orchestrator_pid)"
    fi
    
    # Start learning engine
    local learning_script="${PROJECT_ROOT}/learning_engine/core_engine.py"
    if [[ -f "$learning_script" ]]; then
        log_info "Starting learning engine..."
        nohup $PYTHON_CMD "$learning_script" > "$LOG_DIR/learning.log" 2>&1 &
        local learning_pid=$!
        echo "$learning_pid" > "$LOG_DIR/learning.pid"
        log_info "Learning engine started (PID: $learning_pid)"
    fi
    
    return 0
}

stop_background_services() {
    log_info "Stopping background services..."
    
    # Stop services by PID files
    local pid_files=("$LOG_DIR/worker.pid" "$LOG_DIR/orchestrator.pid" "$LOG_DIR/learning.pid" "$LOG_DIR/redis.pid")
    
    for pid_file in "${pid_files[@]}"; do
        if [[ -f "$pid_file" ]]; then
            local pid
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                local service_name
                service_name=$(basename "$pid_file" .pid)
                log_info "Stopping $service_name (PID: $pid)"
                kill -TERM "$pid" 2>/dev/null || true
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
            rm -f "$pid_file"
        fi
    done
}

# ===============================================================================
# HEALTH CHECK FUNCTIONS
# ===============================================================================

perform_health_checks() {
    log_info "Performing system health checks..."
    
    local health_status=0
    
    # Check Python environment
    if ! $PYTHON_CMD -c "import sys; print(f'Python {sys.version}')"; then
        log_error "Python environment check failed"
        health_status=1
    fi
    
    # Check database connectivity
    local db_check_script="${PROJECT_ROOT}/scripts/check_db.py"
    if [[ -f "$db_check_script" ]]; then
        if ! $PYTHON_CMD "$db_check_script"; then
            log_error "Database connectivity check failed"
            health_status=1
        fi
    else
        log_warn "Database check script not found"
    fi
    
    # Check Redis connectivity
    if command -v redis-cli &> /dev/null; then
        if ! redis-cli ping | grep -q "PONG"; then
            log_warn "Redis connectivity check failed"
        fi
    fi
    
    # Check disk space
    local available_space
    available_space=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/[^0-9.]//g')
    if [[ -n "$available_space" ]] && (( $(echo "$available_space < 1" | bc -l) )); then
        log_warn "Low disk space: ${available_space}GB available"
    fi
    
    # Check memory usage
    if command -v free &> /dev/null; then
        local memory_usage
        memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        if (( $(echo "$memory_usage > 90" | bc -l) )); then
            log_warn "High memory usage: ${memory_usage}%"
        fi
    fi
    
    return $health_status
}

# ===============================================================================
# APPLICATION STARTUP FUNCTIONS
# ===============================================================================

start_ymera_application() {
    log_info "Starting YMERA application server..."
    
    local main_script="${PROJECT_ROOT}/main.py"
    if [[ ! -f "$main_script" ]]; then
        log_error "Main application script not found: $main_script"
        return 1
    fi
    
    # Build uvicorn command
    local uvicorn_cmd=(
        "$PYTHON_CMD" "-m" "uvicorn"
        "main:app"
        "--host" "$HOST"
        "--port" "$PORT"
        "--workers" "$WORKERS"
        "--timeout-keep-alive" "$KEEP_ALIVE"
        "--access-log"
        "--log-level" "info"
    )
    
    # Add environment-specific options
    if [[ "$ENVIRONMENT" == "development" ]]; then
        uvicorn_cmd+=(--reload --reload-dir "$PROJECT_ROOT")
    fi
    
    if [[ "$RUNTIME_ENVIRONMENT" == "replit" ]]; then
        uvicorn_cmd+=(--proxy-headers --forwarded-allow-ips="*")
    fi
    
    log_info "Starting server: ${uvicorn_cmd[*]}"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Start the application
    exec "${uvicorn_cmd[@]}" &
    local app_pid=$!
    export YMERA_PID="$app_pid"
    
    log_info "YMERA application started (PID: $app_pid)"
    
    # Wait for application to be ready
    wait_for_service "YMERA Application" "$HOST" "$PORT" 60 || {
        log_error "YMERA application failed to start"
        return 1
    }
    
    return 0
}

display_startup_summary() {
    log_info "=== YMERA Platform Startup Summary ==="
    log_info "Environment: $RUNTIME_ENVIRONMENT"
    log_info "Python: $PYTHON_CMD"
    log_info "Host: $HOST"
    log_info "Port: $PORT"
    log_info "Workers: $WORKERS"
    log_info "Database: $DATABASE_URL"
    log_info "Redis: $REDIS_URL"
    log_info "Logs: $LOG_DIR"
    log_info "=========================================="
    
    if [[ "$RUNTIME_ENVIRONMENT" == "replit" ]]; then
        log_info "🚀 YMERA is running on Replit!"
        log_info "   Access your application at: https://${REPL_SLUG}.${REPL_OWNER}.repl.co"
    else
        log_info "🚀 YMERA is running!"
        log_info "   Access your application at: http://$HOST:$PORT"
    fi
    
    log_info "📊 Health checks: http://$HOST:$PORT/health"
    log_info "📚 API docs: http://$HOST:$PORT/docs"
    log_info "📋 Admin panel: http://$HOST:$PORT/admin"
    log_info "=========================================="
}

# ===============================================================================
# SIGNAL HANDLERS
# ===============================================================================

setup_signal_handlers() {
    trap cleanup_on_exit EXIT
    trap 'log_info "Received SIGINT, shutting down..."; cleanup_on_exit' INT
    trap 'log_info "Received SIGTERM, shutting down..."; cleanup_on_exit' TERM
}

# ===============================================================================
# MAIN EXECUTION FLOW
# ===============================================================================

main() {
    # Initialize logging
    setup_logging
    
    log_info "=========================================="
    log_info "YMERA Enterprise Platform Startup"
    log_info "Version: $SCRIPT_VERSION"
    log_info "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    log_info "=========================================="
    
    # Setup signal handlers
    setup_signal_handlers
    
    # Environment detection and setup
    detect_environment || {
        log_error "Environment detection failed"
        exit 1
    }
    
    # Python environment setup
    setup_python_environment || {
        log_error "Python environment setup failed"
        exit 1
    }
    
    # Install dependencies
    install_dependencies || {
        log_error "Dependency installation failed"
        exit 1
    }
    
    # Start database services
    start_database_services || {
        log_error "Database services startup failed"
        exit 1
    }
    
    # Setup database
    setup_database || {
        log_error "Database setup failed"
        exit 1
    }
    
    # Start background services
    start_background_services || {
        log_warn "Some background services failed to start"
    }
    
    # Perform health checks
    perform_health_checks || {
        log_warn "Some health checks failed"
    }
    
    # Start main application
    start_ymera_application || {
        log_error "YMERA application startup failed"
        exit 1
    }
    
    # Display startup summary
    display_startup_summary
    
    # Keep script running
    if [[ "${1:-}" != "--no-wait" ]]; then
        log_info "Startup completed successfully. Press Ctrl+C to stop."
        wait "$YMERA_PID"
    fi
}

# ===============================================================================
# SCRIPT EXECUTION
# ===============================================================================

# Validate script is not being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
else
    log_error "This script should be executed, not sourced"
    exit 1
fi