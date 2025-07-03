from llama_cpp import Optional
from pydantic import BaseModel, Field


class ProfileResponse(BaseModel):
    """
    All informations relative to the owner,candidate, of the resume.
    """

    prof_name: Optional[str] = Field(
        default=None, description="String containing first name of the candidate."
    )
    prof_lastname: Optional[str] = Field(
        default=None, description="String containing last name of the candidate."
    )
    prof_date_of_birth: Optional[str] = Field(
        default=None,
        description="String containing date of birth of the candidate. Format it like YYYY-MM-DD.If no date is available, leave it blank.",
    )
    prof_gender: Optional[str] = Field(
        default=None, description="String containing gender of the candidate."
    )
    prof_email: Optional[str] = Field(
        default=None, description="String containing email of the candidate."
    )
    prof_home_phone: Optional[str] = Field(
        default=None, description="String containing home phone of the candidate."
    )
    prof_mobile_phone: Optional[str] = Field(
        default=None, description="String containing mobile phone of the candidate."
    )
    prof_country_code: Optional[str] = Field(
        default=None,
        description="String containing the country of the candidate. Return the standard ISO country code for the given country name (e.g., 'United States' as 'US', 'Canada' as 'CA').",
    )
    prof_city: Optional[str] = Field(
        default=None, description="String containing city of the candidate."
    )
    prof_state: Optional[str] = Field(
        default=None, description="String containing region or state of the candidate."
    )
    prof_address: Optional[str] = Field(
        default=None, description="String containing address of the candidate."
    )
    prof_postcode: Optional[str] = Field(
        default=None, description="String containing postal code of the candidate."
    )
    prof_has_managed_others: Optional[bool] = Field(
        default=None,
        description="boolean indicating if the candidate has managed others.",
    )
    prof_drivers_license: Optional[list[Optional[str]]] = Field(
        default=[], description="Driver licenses of the candidate."
    )
    prof_social_media_links: Optional[list[Optional[str]]] = Field(
        default=[],
        description="Social media http links of the candidate.Format must be HTTP link.",
    )
    prof_salary: Optional[str] = Field(
        default=None,
        description="String containing salary information of the candidate",
    )


class SkillsResponse(BaseModel):
    """
    All informations relative to the skills of the candidate.
    """

    sk_soft_skills: list[Optional[str]] = Field(
        default=[],
        description="List of soft skills extracted from the resume, potentially found in the summary or experience sections. Output a maximum of 10 soft_skills",
    )
    sk_computer_skills: list[Optional[str]] = Field(
        default=[],
        description=(
            "List of computer and technology-related skills ONLY. Include programming languages, "
            "software, frameworks, databases, operating systems, and other technical tools. "
            "DO NOT include soft skills or non-technical skills like 'leadership' or 'patient care'. "
            "Examples of valid skills: Python, Java, AWS, Linux, SQL, Docker, MS Office, Excel, Outlook "
            "Output a maximum of 10 computer_skills."
        ),
    )
    sk_user_skills: list[Optional[str]] = Field(
        default=[],
        description=(
            "List of ALL skills that appear EXPLICITLY in the 'Skills' or 'Technical Skills' section of the resume. "
            "Include every skill listed in these sections, whether technical, soft, or any other type. "
            "These skills can and should overlap with computer_skills and soft_skills if they appear in the skills section. "
            "DO NOT extract skills from other sections like experience or summary. "
            "If there is no dedicated skills section, return an empty list. "
            "Output a maximum of 10 user_skills."
        ),
    )
    sk_summambitsec: Optional[str] = Field(
        default=None, description="Summary or ambition statement for the candidate."
    )
    sk_languages: Optional[list[Optional[str]]] = Field(
        default=[],
        description=(
            "List of languages the candidate can speak. Always include the language the resume is written in (e.g., include "
            "'English' if the resume is in English), plus any additional languages mentioned in the resume."
        ),
    )
