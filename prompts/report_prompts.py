"""
Report Generation Prompts
=========================
Prompt used by generate_report_node to turn raw analyst findings into the
final polished Markdown report shown to the user.

Edit get_report_prompt() to change report structure, section headings, or
formatting requirements.
"""


def get_report_prompt(
    missing_repeating_text: str,
    dataset_summary: str,
    final_analysis: str,
) -> str:
    """
    Returns the prompt that instructs the LLM to write the final Data Quality Report.

    Args:
        missing_repeating_text: Column-wise null/missing percentages block.
        dataset_summary:        High-level dataset check summaries (Group 1 overview).
        final_analysis:         Raw findings text produced by the Quality Analyst.
    """
    return f"""Based on the following data quality analysis findings, write a clear, professional Data Quality Report.
    The report should have sections for:
    1. Executive Summary
    2. Missing and Repeating Values
       IMPORTANT: You MUST explicitly include the following column-wise percentages for ALL columns in the dataset:
{missing_repeating_text}
    3. Advanced Dataset Quality
       Include these high-level dataset metrics:
{dataset_summary}
    4. Logical Inconsistencies and Invalid Values
    5. Recommendations
    
    IMPORTANT: For section "4. Logical Inconsistencies and Invalid Values", you MUST present the findings as a Markdown table. The table should have exactly two columns: "Variable Name" and "Inconsistencies Found". Do not use a numbered list for this section.
    
    Analysis Findings:
    {final_analysis}
    """
