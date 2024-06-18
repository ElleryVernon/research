from openai import OpenAI
from textwrap import dedent
import os

os.environ["OPENAI_API_KEY"] = "apikey"
client = OpenAI()


class ResearchAssistant:
    def __init__(self, model: str = "gpt-4o"):
        """Initialize the Research Assistant with a specific model."""
        self.model = model
        self.instructions = dedent(
            """
            You are a Senior NYT Editor tasked with writing a NYT cover story worthy report due tomorrow.
            You will be provided with a topic and search results from junior researchers.
            Carefully read the results and generate a final - NYT cover story worthy report.
            Make your report engaging, informative, and well-structured.
            Your report should follow the format provided below.
            Remember: you are writing for the New York Times, so the quality of the report is important.

            <report_format>
            ## Title of report

            - **Overview** Brief introduction of the topic.
            - **Importance** Why is this topic significant now?

            ### Section 1
            - **Detail 1**
            - **Detail 2**
            - **Detail 3**

            ### Section 2
            - **Detail 1**
            - **Detail 2**
            - **Detail 3**

            ### Section 3
            - **Detail 1**
            - **Detail 2**
            - **Detail 3**

            ## Conclusion
            - **Summary of report:** Recap of the key findings from the report.
            - **Implications:** What these findings mean for the future.

            ## References
            - [Reference 1](Link to Source)
            - [Reference 2](Link to Source)
            </report_format>
            """
        )

    def generate_report(self, topic: str, search_results: str) -> str:
        """Generate a report based on the topic and search results."""
        prompt = f"Topic: {topic}\n\nSearch Results:\n{search_results}\n\nPlease generate the report in the specified format."

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0.05,
            presence_penalty=0.1,
        )

        return response.choices[0].message.content
