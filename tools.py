import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")
ADZUNA_LOCATION = os.getenv("ADZUNA_LOCATION", "India")

from langchain_core.tools import tool
from retriever import retrieve_course_material


@tool
def retrieve_course_material_tool(question: str) -> str:
    """Retrieve relevant course notes to answer questions about studied concepts."""

    try:
        results = retrieve_course_material(question, k=3)

        if not results:
            return "No relevant course material was found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            formatted_results.append(
                f"Passage {index}:\n"
                f"Source: {result['source']}\n"
                f"Content: {result['content']}"
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        return f"Course material retrieval failed: {error}"
@tool
def search_jobs_tool(
    topic: str,
    location: str = "India",
    minimum_salary: int = 0,
    maximum_salary: int = 0,
) -> str:
    """
    Search live Adzuna jobs and internships by topic and location.

    Args:
        topic: Job title, skill, or topic, such as Python internship.
        location: City or region, such as Ahmedabad, Gujarat, or India.
        minimum_salary: Minimum annual salary in local currency. Use 0 for no minimum.
        maximum_salary: Maximum annual salary in local currency. Use 0 for no maximum.
    """
    import os
    from pathlib import Path

    import requests
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env", override=True)

    if not topic.strip():
        return "Please provide a job topic or skill."

    if minimum_salary < 0 or maximum_salary < 0:
        return "Salary filters cannot be negative."

    if maximum_salary and minimum_salary > maximum_salary:
        return "Minimum salary cannot exceed maximum salary."

    app_id = os.getenv("ADZUNA_APP_ID", "").strip()
    app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
    country = os.getenv("ADZUNA_COUNTRY", "in").strip()

    if not app_id or not app_key:
        return "Adzuna credentials are missing. Check your .env file."

    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": topic.strip(),
        "where": location.strip(),
        "results_per_page": 10,
        "content-type": "application/json",
        "sort_by": "relevance",
    }

    if minimum_salary > 0:
        params["salary_min"] = minimum_salary

    if maximum_salary > 0:
        params["salary_max"] = maximum_salary

    try:
        response = requests.get(
            url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        jobs = data.get("results", [])

        if not jobs:
            return (
                f"No jobs found for '{topic}' in '{location}' "
                "with the requested salary filters. Try broadening your search."
            )

        formatted_jobs = []

        for index, job in enumerate(jobs, start=1):
            title = job.get("title", "Not specified")
            company = job.get("company", {}).get(
                "display_name", "Not specified"
            )
            job_location = job.get("location", {}).get(
                "display_name", "Not specified"
            )
            description = job.get(
                "description", "No description provided."
            )
            apply_url = job.get("redirect_url", "No application link available")

            salary_min = job.get("salary_min")
            salary_max = job.get("salary_max")
            currency = "INR" if country == "in" else "local currency"

            if salary_min is None and salary_max is None:
                salary_text = "Not disclosed"
            elif salary_min is not None and salary_max is not None:
                salary_text = (
                    f"{salary_min:,.0f}–{salary_max:,.0f} {currency} per year"
                )
            elif salary_min is not None:
                salary_text = f"From {salary_min:,.0f} {currency} per year"
            else:
                salary_text = f"Up to {salary_max:,.0f} {currency} per year"

            formatted_jobs.append(
                f"JOB {index}: {title}\n"
                f"Company: {company}\n"
                f"Location: {job_location}\n"
                f"Minimum salary: "
                f"{salary_min if salary_min is not None else 'Not disclosed'}\n"
                f"Maximum salary: "
                f"{salary_max if salary_max is not None else 'Not disclosed'}\n"
                f"Salary range: {salary_text}\n"
                f"Description: {description[:1000]}\n"
                f"Apply: {apply_url}"
            )

        return (
            f"Live Adzuna results for '{topic}' in '{location}'. "
            f"Results returned: {len(jobs)}.\n\n"
            + "\n\n".join(formatted_jobs)
        )

    except requests.RequestException as error:
        status = (
            f"HTTP {error.response.status_code}"
            if error.response is not None
            else "Connection error"
        )
        return f"Adzuna search failed ({status}). Please try again."
    except (ValueError, TypeError):
        return "Adzuna returned an unexpected response."