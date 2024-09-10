# output_parser.py

from langchain.schema import BaseOutputParser
import re

class LastPartOutputParser(BaseOutputParser):
    def parse(self, output: str) -> str:
        # Split the output by 'answer:' to find the last relevant part
        parts = output.split("answer:")
        if len(parts) > 1:
            # The last part is the valid answer
            final_answer = parts[-1].strip()
        else:
            # If there is no 'answer:', return the whole output
            final_answer = output.strip()

        # Replace '\n' with '  \n' in the final answer
        formatted_answer = final_answer.replace('\n', '  \n')

        # Use regex to find and replace the metadata pattern
        formatted_answer = self._format_metadata(formatted_answer)
        
        return formatted_answer

    def _format_metadata(self, text: str) -> str:
        # Regex pattern to match the metadata format and exclude './uploads/'
        metadata_pattern = r"\(source:\s*\.\/uploads\/([^|]+)\|\s*page:\s*(\d+)\)"
        
        # Function to replace the matched pattern with the desired format
        def replace_metadata(match):
            filename = match.group(1).strip()
            page_number = match.group(2).strip()
            return f"(출처 : {filename} | page: {page_number})"
        
        # Replace all occurrences in the text
        return re.sub(metadata_pattern, replace_metadata, text)
