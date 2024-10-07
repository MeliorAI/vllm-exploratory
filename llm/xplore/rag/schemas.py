from datetime import datetime
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, field_validator, validator

from llm.xplore import get_logger

logger = get_logger(__name__)


# ============= Dynamic Insights Models ==============
class QuestionType(str, Enum):
    YES_NO = "yes/no"
    SUMMARISATION = "summarisation"
    DATETIME = "datetime"
    EXTRACTIVE = "extractive"
    CATEGORICAL = "categorical"


class QuestionDetail(BaseModel):
    question: str
    question_id: str
    alias: str
    question_type: QuestionType


class QuestionTypeRequest(BaseModel):
    question: str
    provided_type: str | None = None


class YesNoEnum(Enum):
    pos = "Yes"
    neg = "No"
    NON = None


class YesNoSchema(BaseModel):
    result: YesNoEnum | None


class DateSchema(BaseModel):
    date: str

    @field_validator("date")
    @classmethod
    def check_date_format(cls, value) -> str | None:
        # List of acceptable date formats
        date_formats = [
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
        ]

        for date_format in date_formats:
            try:
                # Try to parse the date using the current format
                parsed_date = datetime.strptime(value, date_format)  # noqa: DTZ007

                # Standardize the format, we can select a different one in the future
                date_format = "%Y-%m-%d %H:%M:%S"
                parsed_date_formatted = datetime.strptime(str(parsed_date), date_format)  # noqa: DTZ007
                return parsed_date_formatted.strftime(date_format)
            except ValueError as e:  # noqa: PERF203
                error = e

        logger.error(f"Error {error}, Date '{value}' is not in a recognized format")
        return None


class SummarySchema(BaseModel):
    summary: str


class QuestionFormat(TypedDict):
    message: str
    schema: type[BaseModel]


question_formats: dict[str, QuestionFormat] = {
    "yes/no": {
        "message": "Tips: Make sure to answer in the correct format: 'Yes' or 'No' ",
        "schema": YesNoSchema,
    },
    "summarisation": {
        "message": "Tips: Make concise summary, if you can't find the answer in the context return None ",
        "schema": SummarySchema,
    },
    "extractive": {
        "message": "Tips: If you can't find the answer in the context return None ",
        "schema": SummarySchema,
    },
    "datetime": {
        "message": "Tips: Make sure to answer in the correct format, dates in the datetime format, if you can't find the answer in the context return None ",
        "schema": DateSchema,
    },
}

# Construct the question types and their descriptions dynamically
question_type_descriptions = {
    QuestionType.YES_NO: "A binary question expecting a 'Yes' or 'No' answer.",
    QuestionType.SUMMARISATION: "A question asking for a brief summary of the provided context.",
    QuestionType.DATETIME: "A question asking for a date and time, which requires a specific format.",
    QuestionType.EXTRACTIVE: "A question requiring extraction of specific information directly from the text.",
    QuestionType.CATEGORICAL: "A question asking to classify something into predefined categories.",
}
