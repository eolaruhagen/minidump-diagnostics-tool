import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from app_logger import logger


class PromptTemplateManager:
    """Manager for all debugging prompt templates following new structure."""
    
    def __init__(self, whitelisted_commands_path: str = "user_docs/whitelisted_commands.json"):
        """Initialize the prompt template manager."""
        self.whitelisted_commands_path = Path(whitelisted_commands_path)
        self._whitelisted_commands = self._load_whitelisted_commands()
        
        # Special commands
        self.stop_command = "STOP_DEBUGGING"
        
        # Authorized bugcheck codes (from new_prompting.md) - support both formats
        self.authorized_bugcheck_codes = [
            '0xA', '0xD1', '0xC4', '0xC9', '0x50', '0x7E', '0x8E', '0x1E',  # with 0x
            'A', 'D1', 'C4', 'C9', '50', '7E', '8E', '1E',                  # without 0x uppercase
            'a', 'd1', 'c4', 'c9', '50', '7e', '8e', '1e', '4a'            # lowercase (including 4a)
        ]
    
    def _load_whitelisted_commands(self) -> List[Dict[str, Any]]:
        """Load whitelisted commands from JSON file."""
        try:
            with open(self.whitelisted_commands_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Whitelisted commands file not found: {self.whitelisted_commands_path}") from exc
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in whitelisted commands file: {e}") from e
    
    def _format_whitelisted_commands(self, exclude_commands: List[str] = None) -> str:
        """Format whitelisted commands for inclusion in prompts with aliases and argument info."""
        if exclude_commands is None:
            exclude_commands = []
            
        # Normalize exclude list to use command names (aliases) - keep original format for exact matching
        exclude_set = set(exclude_commands)
        
        if exclude_commands:
            logger.info(f"Excluding commands: {exclude_commands}")
        
        commands_text = []
        for cmd in self._whitelisted_commands:
            # Check if this command should be excluded (by exact name/alias match)
            if cmd['name'] in exclude_set:
                logger.debug(f"Excluding used command: {cmd['name']}")
                continue
            
            # Format command with alias and argument requirement
            arg_info = " (requires argument)" if cmd.get('has_arguments', False) else " (no argument)"
            cmd_text = f"- {cmd['name']}{arg_info}: {cmd['purpose']}"
            commands_text.append(cmd_text)
        
        logger.info(f"Available commands after exclusions: {len(commands_text)} out of {len(self._whitelisted_commands)}")
        return "\n".join(commands_text)
    
    def get_command_count(self) -> int:
        """Get the total number of whitelisted commands."""
        return len(self._whitelisted_commands)
    
    def get_overall_sysprompt(self) -> str:
        """Get the overall system prompt (everything but final)."""
        return """You are an expert Windows kernel debugging specialist. Your role is to systematically analyze crash dumps using WinDbg commands and provide structured analysis with proper citations.
        Specifically, you will want to identify the root cause of the crash under the assumption that the user already is aware of the bugcheck code, so focus on deeper analysis of specific citations that you will create.
        The final goal is to identify the root cause of the crash with high confidence, and provide a detailed explanation of how you arrived at that conclusion, creating analysis and explanations of kernel level systems that less experienced users can learn from.
        You will use the available commands to gather information, and you will cite specific lines of output.

## CRITICAL FORMATTING REQUIREMENTS:

You MUST use these exact delimiters to separate sections of your response:

### Citations Section:
Start with: <CITATIONS>
End with: </CITATIONS>
Format inside: [value_type] : [true_value] : short 1-2 sentence analysis
Example:
<CITATIONS>
[BUGCHECK_CODE] : [0x4a] : Indicates an IRQL violation by a driver....
[FAULT_ADDRESS] : [0x12345678] : Shows memory corruption in driver space....
</CITATIONS>

### Command Section:
Start with: <COMMAND>
End with: </COMMAND>
Format inside: command_alias (the name from available commands list)
Example:
<COMMAND>
driver_object_info
</COMMAND>

### Command Argument Section:
Start with: <CMD_ARGUMENT>
End with: </CMD_ARGUMENT>
Format inside: argument value (use "NO_ARGUMENT" if command takes no argument)
Example:
<CMD_ARGUMENT>
mydriver.sys
</CMD_ARGUMENT>

ALWAYS use these delimiters exactly as shown. Your response will be parsed automatically using these markers."""
    
    def get_main_loop_sysprompt(self) -> str:
        """Get the system prompt specifically for main loop (includes RAG query)."""
        return """You are an expert Windows kernel debugging specialist. Your role is to systematically analyze crash dumps using WinDbg commands and provide structured analysis with proper citations.
        Specifically, you will want to identify the root cause of the crash under the assumption that the user already is aware of the bugcheck code, so focus on deeper analysis of specific citations that you will create.

## CRITICAL FORMATTING REQUIREMENTS:

You MUST use these exact delimiters to separate sections of your response:

### Citations Section:
Start with: <CITATIONS>
End with: </CITATIONS>
Format inside: [value_type] : [true_value] : short 1-2 sentence analysis
Example:
<CITATIONS>
[BUGCHECK_CODE] : [0x4a] : Indicates an IRQL violation by a driver....
[FAULT_ADDRESS] : [0x12345678] : Shows memory corruption in driver space....
</CITATIONS>

### Command Section:
Start with: <COMMAND>
End with: </COMMAND>
Format inside: command_alias (the name from available commands list)
Example:
<COMMAND>
driver_object_info
</COMMAND>

### Command Argument Section:
Start with: <CMD_ARGUMENT>
End with: </CMD_ARGUMENT>
Format inside: argument value (use "NO_ARGUMENT" if command takes no argument)
Example:
<CMD_ARGUMENT>
mydriver.sys
</CMD_ARGUMENT>

### RAG Query Section:
Start with: <RAG_QUERY>
End with: </RAG_QUERY>
Format inside: Natural language query for context retrieval, this is NOT a query to an LLM, but rather a search query for your available context in the vector store.
Therefore, keep it concise, and maximize relevance to the debugging context with important keywords.
Example:
<RAG_QUERY>
driver IRQL violation memory access fault address stack frame ntoskrnl
</RAG_QUERY>

ALWAYS use these delimiters exactly as shown. Your response will be parsed automatically using these markers."""
    
    def get_initial_sysprompt(self) -> str:
        """Get the initial system prompt with authorized bugcheck codes. Note that The line BUGCHECK_CODE may output variations of a code for example (0x4a, 4a, 4A, etc.) ONLY PAY ATTENTION TO THE CODE ON THE LINE WITH BUGCHECK_CODE"""
        # Show only the main codes in a clean format
        main_codes = "0x4A, 0xA, 0xD1, 0xC4, 0xC9, 0x50, 0x7E, 0x8E, 0x1E"
        return f"""You are authorized to analyze only these bugcheck codes: {main_codes}

If the crash has a different bugcheck code, respond with exactly "STOP_DEBUGGING" as the command and note that this crash type is not compatible."""
    
    def get_initial_prompt(self, analyze_v_output: str) -> str:
        """Create the initial prompt with full analyze -v output and commands."""
        commands_list = self._format_whitelisted_commands()
        
        return f"""{self.get_overall_sysprompt()}

{self.get_initial_sysprompt()}

## CRASH DUMP ANALYSIS:
```
{analyze_v_output}
```

## AVAILABLE COMMANDS:
{commands_list}

## YOUR TASK:
1. First check if the bugcheck code is in the authorized list
2. If not authorized, respond with STOP_DEBUGGING command only
3. If authorized, provide citations from the crash dump in the required delimited format
4. Then provide the next command alias and argument using the required delimited format

Provide your response using the exact delimiter format specified above."""
    
    def get_main_loop_prompt(self, 
                           analyze_v_output: str,
                           command_outputs: List[Tuple[str, str]],  # (command, output)
                           rag_context: str = None,
                           used_commands: List[str] = None) -> str:
        """Create main loop prompt with context and available commands."""
        
        if used_commands is None:
            used_commands = []
        
        # Create current context
        context_sections = [f"## INITIAL ANALYZE -V OUTPUT:\n```\n{analyze_v_output}\n```"]
        
        # Add previous command outputs
        if command_outputs:
            context_sections.append("## PREVIOUS COMMAND OUTPUTS:")
            for command, output in command_outputs:
                context_sections.append(f"### Command: {command}\n```\n{output}\n```")
        
        # Add RAG context only if provided and not empty
        if rag_context and rag_context.strip():
            context_sections.append(f"## RETRIEVED DEBUGGING CONTEXT:\n{rag_context}")
        
        current_context = "\n\n".join(context_sections)
        
        # Get available commands (excluding used ones)
        available_commands = self._format_whitelisted_commands(exclude_commands=used_commands)
        
        return f"""{self.get_main_loop_sysprompt()}

## CURRENT CONTEXT:
{current_context}

## AVAILABLE COMMANDS:
{available_commands}

## YOUR TASK:
Based on the current context, provide:
1. Citations from the debugging data using <CITATIONS></CITATIONS> delimiters
2. Next command alias to execute (or STOP_DEBUGGING if you have definitive root cause) using <COMMAND></COMMAND> delimiters
3. Command argument (or NO_ARGUMENT if none needed) using <CMD_ARGUMENT></CMD_ARGUMENT> delimiters
4. RAG query for retrieving relevant context using <RAG_QUERY></RAG_QUERY> delimiters

You MUST use the exact delimiter format specified in the system prompt above."""
    
    def get_final_prompt(self, all_citations: List[str], analyze_v_output: str) -> str:
        """Create final prompt for aggregating all citations into JSON output."""
        # Since all_citations now contains individual citations, number them properly
        citations_context = "\n".join([f"Citation {i+1}: {citation}" for i, citation in enumerate(all_citations)])
        
        return f"""You are an expert Windows kernel debugging specialist. Your task is to provide a final comprehensive analysis.

## FINAL ANALYSIS TASK:
You have completed the debugging process. Now aggregate ALL citations and provide a final analysis ***ALL CITATIONS PROVIDED HERE MUST ALSO BE PRESENT IN THE FINAL JSON OUTPUT***. The final goal is to identify the root cause of the crash with high confidence, and provide a detailed explanation of how you arrived at that conclusion, creating analysis and explanations of kernel level systems that less experienced users can learn from. You will use the available commands to gather information, and you will cite specific lines of output.

## INITIAL ANALYZE -V OUTPUT:
```
{analyze_v_output}
```

## ALL CITATIONS COLLECTED ({len(all_citations)} total):
{citations_context}

## OUTPUT FORMAT:
Provide your response in this exact format with two separate sections:

**SECTION 1 - JSON CITATIONS:**
CRITICAL: You must provide exactly {len(all_citations)} citations in the JSON array (one for each citation I provided above).
Provide a JSON object with just the citations (no analysis field):
```json
{{
    "citations": [
        {{
            "value_name": "BUGCHECK_CODE",
            "description": "Brief description of this citation"
        }},
        {{
            "value_name": "PROCESS_NAME", 
            "description": "Brief description of this citation"
        }}
    ]
}}
```

**SECTION 2 - ANALYSIS TEXT:**
After the JSON, provide a comprehensive narrative analysis as plain text (not JSON):

Understanding the Problem:
[Your detailed explanation of what went wrong]

Technical Details:
[Technical analysis of the crash data and evidence]

Root Cause Analysis:
[Your investigation methodology and reasoning]

Conclusion:
[Final determination with confidence level (Low/Medium/High)]

IMPORTANT: 
- You MUST include exactly {len(all_citations)} citations in the JSON array - one for each citation provided above
- Aggregate and improve ALL OF THE GIVEN citations in the JSON section  
- Make sure every significant finding has a corresponding citation
- The analysis should reference the citations but be written as flowing narrative text
- Do not put the analysis in JSON format - it should be plain text after the JSON

REMINDER: Your citations array must contain {len(all_citations)} items - no more, no less."""
    
    def extract_citations_from_response(self, response: str) -> str:
        """Extract citations from LLM response using <CITATIONS></CITATIONS> delimiters."""
        # Find content between <CITATIONS> and </CITATIONS>
        citations_pattern = r'<CITATIONS>(.*?)</CITATIONS>'
        match = re.search(citations_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            citations_content = match.group(1).strip()
            # Clean up and validate citation format
            citation_lines = []
            for line in citations_content.split('\n'):
                line = line.strip()
                if line and re.match(r'^\[.*?\]\s*:\s*.+', line):
                    citation_lines.append(line)
            return "\n".join(citation_lines)
        
        # Fallback to old method if delimiters not found
        logger.debug("Citations delimiters not found, using fallback method")
        lines = response.split('\n')
        citations = []
        
        for line in lines:
            line = line.strip()
            # Look for citation pattern [VALUE] : analysis
            if re.match(r'^\[.*?\]\s*:\s*.+', line):
                citations.append(line)
        
        return "\n".join(citations)
    
    def extract_individual_citations_from_response(self, response: str) -> List[str]:
        """
        Extract individual citations from LLM response as a list.
        This properly separates multiple citations instead of returning them as one block.
        """
        # Find content between <CITATIONS> and </CITATIONS>
        citations_pattern = r'<CITATIONS>(.*?)</CITATIONS>'
        match = re.search(citations_pattern, response, re.DOTALL | re.IGNORECASE)
        
        individual_citations = []
        
        if match:
            citations_content = match.group(1).strip()
            # Parse each line as a separate citation
            for line in citations_content.split('\n'):
                line = line.strip()
                if line and re.match(r'^\[.*?\]\s*:\s*.+', line):
                    individual_citations.append(line)
        else:
            # Fallback to old method if delimiters not found
            logger.debug("Citations delimiters not found, using fallback method")
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for citation pattern [VALUE] : analysis
                if re.match(r'^\[.*?\]\s*:\s*.+', line):
                    individual_citations.append(line)
        
        return individual_citations
    
    def extract_command_from_response(self, response: str) -> str:
        """Extract the command alias from LLM response using <COMMAND></COMMAND> delimiters."""
        # Find content between <COMMAND> and </COMMAND>
        command_pattern = r'<COMMAND>(.*?)</COMMAND>'
        match = re.search(command_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            command_content = match.group(1).strip()
            # Return the first non-empty line
            for line in command_content.split('\n'):
                line = line.strip()
                if line:
                    return line
        
        # Fallback to old method if delimiters not found
        logger.debug("Command delimiters not found, using fallback method")
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for command patterns
            if line.startswith('!') or line.startswith('k') or line.startswith('lm'):
                return line
            # Check for STOP_DEBUGGING
            if line == self.stop_command:
                return line
        
        return ""
    
    def extract_command_argument_from_response(self, response: str) -> str:
        """Extract the command argument from LLM response using <CMD_ARGUMENT></CMD_ARGUMENT> delimiters."""
        # Find content between <CMD_ARGUMENT> and </CMD_ARGUMENT>
        arg_pattern = r'<CMD_ARGUMENT>(.*?)</CMD_ARGUMENT>'
        match = re.search(arg_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            arg_content = match.group(1).strip()
            # Return the first non-empty line
            for line in arg_content.split('\n'):
                line = line.strip()
                if line:
                    return line
        
        return "NO_ARGUMENT"
    
    def extract_rag_query_from_response(self, response: str) -> str:
        """Extract RAG query from LLM response using <RAG_QUERY></RAG_QUERY> delimiters."""
        # Find content between <RAG_QUERY> and </RAG_QUERY>
        rag_query_pattern = r'<RAG_QUERY>(.*?)</RAG_QUERY>'
        match = re.search(rag_query_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            rag_query_content = match.group(1).strip()
            # Return the first non-empty line
            for line in rag_query_content.split('\n'):
                line = line.strip()
                if line:
                    return line
        
        # Simple fallback without warnings or complex search
        return "debugging context"
    
    def validate_response_format(self, response: str, expected_sections: List[str]) -> Dict[str, bool]:
        """
        Validate that the LLM response contains all expected delimiter sections.
        
        Args:
            response: The LLM response text
            expected_sections: List of section names to check (e.g., ['CITATIONS', 'COMMAND', 'RAG_QUERY'])
        
        Returns:
            Dictionary mapping section names to whether they were found
        """
        validation_results = {}
        
        for section in expected_sections:
            pattern = f'<{section}>(.*?)</{section}>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            validation_results[section] = match is not None
            
        return validation_results
    
    def validate_command_arguments(self, command_alias: str, argument: str, vector_manager) -> bool:
        """Validate command arguments according to new spec using command alias and argument."""
        if not command_alias or command_alias == self.stop_command:
            return True
        
        # Find the command definition by alias (name)
        command_info = None
        for cmd in self._whitelisted_commands:
            if cmd['name'] == command_alias:
                command_info = cmd
                break
        
        if not command_info:
            logger.warning(f"Command alias '{command_alias}' not found in whitelisted commands")
            return False
        
        # Check if command requires arguments
        requires_args = command_info.get('has_arguments', False)
        
        if requires_args and argument == "NO_ARGUMENT":
            logger.warning(f"Command '{command_alias}' requires an argument but NO_ARGUMENT was provided")
            return False
        
        if not requires_args and argument != "NO_ARGUMENT":
            logger.warning(f"Command '{command_alias}' does not take arguments but '{argument}' was provided")
            return False
        
        # If no argument is needed, validation passes
        if not requires_args:
            return True
        
        # Validate the argument exists in vector store using document search
        logger.info(f"Validating argument '{argument}' for command '{command_alias}'")
        
        try:
            # Search through all documents to see if the argument exists
            for doc_id in vector_manager.document_uuids:
                try:
                    doc = vector_manager.vectorstore.get_by_ids([doc_id])[0]
                    if argument.lower() in doc.page_content.lower():
                        logger.info(f"Argument '{argument}' found in document {doc_id}")
                        return True
                except (IndexError, AttributeError):
                    continue
            
            logger.warning(f"Argument '{argument}' not found in any vector store documents")
            return False
            
        except (AttributeError, ValueError, RuntimeError) as e:
            logger.error(f"Error during argument validation: {e}")
            return True  # Assume valid if search fails
    
    def reconstruct_command(self, command_alias: str, argument: str) -> str:
        """Reconstruct the actual WinDbg command from alias and argument."""
        if command_alias == self.stop_command:
            return self.stop_command
        
        # Find the command definition by alias (name)
        command_info = None
        for cmd in self._whitelisted_commands:
            if cmd['name'] == command_alias:
                command_info = cmd
                break
        
        if not command_info:
            logger.error(f"Cannot reconstruct command: alias '{command_alias}' not found")
            return ""
        
        command_template = command_info['command']
        
        # If the command has arguments, replace placeholders
        if command_info.get('has_arguments', False) and argument != "NO_ARGUMENT":
            # Replace common placeholder patterns
            if '<driver>' in command_template:
                command_template = command_template.replace('<driver>', argument)
            elif '<module>' in command_template:
                command_template = command_template.replace('<module>', argument)
            elif '<P1>' in command_template:
                command_template = command_template.replace('<P1>', argument)
            else:
                # If no specific placeholder, append the argument
                command_template = f"{command_template} {argument}"
        
        return command_template    
    def is_stop_command(self, command: str) -> bool:
        """Check if command is STOP_DEBUGGING."""
        return command.strip() == self.stop_command
    
    def is_unauthorized_bugcheck(self, analyze_v_output: str) -> bool:
        """Check if the bugcheck code is not in authorized list."""
        # Look for bugcheck code in the actual format: BUGCHECK_CODE: 4a
        bugcheck_pattern = r'BUGCHECK_CODE:\s*([A-Fa-f0-9]+)'
        match = re.search(bugcheck_pattern, analyze_v_output)
        
        if match:
            bugcheck_code = match.group(1)
            # Check if this code is in our authorized list (any format)
            return bugcheck_code not in self.authorized_bugcheck_codes
        
        # Fallback: look for other bugcheck patterns
        fallback_patterns = [
            r'BUGCHECK_STR:\s*(0x[A-Fa-f0-9]+)',  # BUGCHECK_STR: 0x4a
            r'Bug[Cc]heck\s+(0x[A-Fa-f0-9]+)',   # Bugcheck 0x4a
            r'Stop:\s*(0x[A-Fa-f0-9]+)'           # Stop: 0x4a
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, analyze_v_output)
            for code in matches:
                # Normalize code (remove 0x, handle case)
                normalized_code = code.replace('0x', '').replace('0X', '')
                if normalized_code in self.authorized_bugcheck_codes:
                    return False
        
        # If no bugcheck code found at all, assume unauthorized to be safe
        return True
    
    def extract_json_from_final_response(self, response: str) -> dict:
        """Extract JSON citations from final response."""
        
        # TEMPORARY: Log the raw final response to debug parsing issues
        logger.info("="*60)
        logger.info("DEBUG: Raw final LLM response:")
        logger.info("="*60)
        logger.info(response)
        logger.info("="*60)
        
        try:
            # Find JSON block in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            logger.info(f"DEBUG: JSON boundaries found - start: {json_start}, end: {json_end}")
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                logger.info("DEBUG: Extracted JSON string:")
                logger.info(json_str)
                
                parsed_json = json.loads(json_str)
                logger.info("DEBUG: JSON parsing successful!")
                
                # Note: No need for complex formatting since we're only parsing citations
                return parsed_json
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"DEBUG: JSON parsing failed with error: {e}")
            logger.error(f"DEBUG: Failed to parse JSON from response of length {len(response)}")
        
        # Fallback: create basic structure
        logger.warning("DEBUG: Using fallback JSON structure")
        return {
            "citations": []
        }
    
    def extract_analysis_text_from_response(self, response: str) -> str:
        """Extract the plain text analysis from final response (after the JSON section)."""
        try:
            # Find the end of the JSON block
            json_end = response.rfind('}')
            
            if json_end != -1:
                # Get everything after the JSON
                text_after_json = response[json_end + 1:].strip()
                
                # Look for "SECTION 2" or analysis content
                analysis_start_patterns = [
                    "**SECTION 2",
                    "ANALYSIS TEXT:",
                    "Understanding the Problem:",
                    "The root cause"
                ]
                
                analysis_text = text_after_json
                for pattern in analysis_start_patterns:
                    if pattern in text_after_json:
                        start_idx = text_after_json.find(pattern)
                        analysis_text = text_after_json[start_idx:].strip()
                        break
                
                # Clean up any markdown formatting
                analysis_text = analysis_text.replace("**SECTION 2 - ANALYSIS TEXT:**", "").strip()
                
                if analysis_text and len(analysis_text) > 50:  # Reasonable minimum length
                    # Apply smart line wrapping with 130 character limit
                    return self._wrap_text_smartly(analysis_text, max_line_length=130)
                    
        except Exception as e:
            logger.error(f"Error extracting analysis text: {e}")
        
        # Fallback
        return "Unable to extract analysis text from LLM response"
    
    def _wrap_text_smartly(self, text: str, max_line_length: int = 130) -> str:
        """
        Wrap text to specified line length, breaking only at word boundaries.
        Preserves paragraphs and section headers.
        """
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Empty lines (paragraph breaks) - preserve them
            if not line:
                wrapped_lines.append('')
                continue
                
            # Short lines (likely headers) - keep as-is if they fit
            if len(line) <= max_line_length:
                wrapped_lines.append(line)
                continue
            
            # Long lines - wrap them intelligently
            words = line.split()
            current_line = ""
            
            for word in words:
                # Test if adding this word would exceed the limit
                test_line = current_line + (" " if current_line else "") + word
                
                if len(test_line) <= max_line_length:
                    current_line = test_line
                else:
                    # Current line is full, save it and start a new one
                    if current_line:
                        wrapped_lines.append(current_line)
                        current_line = word
                    else:
                        # Single word is longer than max_line_length
                        # Keep it as-is rather than breaking mid-word
                        wrapped_lines.append(word)
                        current_line = ""
            
            # Add any remaining text
            if current_line:
                wrapped_lines.append(current_line)
        
        return '\n'.join(wrapped_lines)
    
    # Note: _clean_json_formatting method removed since we now use separate text files for analysis