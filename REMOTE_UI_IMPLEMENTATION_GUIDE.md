# UI Implementation Guide: Remote Tool Bridge

This document explains how to update the **UI Application** to support the "Antigravity Bridge." This allows your hosted server to safely access local files and OCR via WebSockets.

### Proactive Project Context (Initial Message)
The UI can now "educate" the brain immediately by sending a `project_context` block in the very first message of a thread.

**Structure of the initial message:**
```json
{
  "message": "Analyze the noise regulations.",
  "project_context": [
    {
      "path": "/project/ordinance_1.pdf",
      "documentType": "Наредба",
      "summary": "Covers noise regulations (14:00-16:00 and 22:00-08:00)...",
      "indexFields": { "Municipality": "Ботевград", "Year": "2019" },
      "parties": [ { "role": "Issuer", "name": "Botevgrad Council" } ]
    }
  ]
}
```

### Incoming Event (`TOOL_CALL`)
```json
{
  "type": "TOOL_CALL",
  "call_id": "uuid-string",
  "tool_name": "read_file",
  "args": {
    "file_path": "/project/contract.pdf"
  }
}
```

### Outgoing Response (`TOOL_RESULT`)
```json
{
  "type": "TOOL_RESULT",
  "call_id": "uuid-string",
  "result": "Full content of the file or OCR result...",
  "error": null
}
```

---

## 2. Required Tools to Implement

The following tools must be handled by your UI's WebSocket listener:

| Tool Name | Action Needed |
| :--- | :--- |
| `read_file` | Read a local text file and return the string. |
| `write_file` | Write text content to `/output/` or `/templates/`. |
| `execute_ocr` | Run your local **PaddleOCR** script on a PDF/Image. |
| `search_indexed` | Scan the local `.indexedfiles` folder for matches. |
| `read_document` | Parse a `.docx` or `.pdf` locally and return the text. |
| `list_directory` | Return a list of files in a specific virtual path. |
| `get_project_overview` | Return an array of all file summaries and metadata (same as `project_context` format). |

---

## 3. Reference Implementation (JavaScript/Node.js)

```javascript
// Example handler for your WebSocket client
socket.on('message', async (data) => {
    const event = JSON.parse(data);

    if (event.type === 'TOOL_CALL') {
        try {
            const result = await handleToolExecution(event.tool_name, event.args);
            
            // Send back the result
            socket.send(JSON.stringify({
                type: 'TOOL_RESULT',
                call_id: event.call_id,
                result: result,
                error: null
            }));
        } catch (err) {
            socket.send(JSON.stringify({
                type: 'TOOL_RESULT',
                call_id: event.call_id,
                result: null,
                error: err.message
            }));
        }
    }
});

async function handleToolExecution(name, args) {
    switch (name) {
        case 'read_file':
            return fs.readFileSync(resolveVirtualPath(args.file_path), 'utf8');
            
        case 'execute_ocr':
            // Call your local python paddle_ocr_script.py
            return execOcrScript(args.file_path);
            
        case 'search_indexed':
            // Port your search logic here or call a local helper
            return localSearch(args.query);
            
        default:
            throw new Error(`Tool ${name} not supported locally.`);
    }
}
```

## 4. Security Considerations
- **HTTPS/WSS**: Always use secure WebSockets (`wss://`) for the bridge.
- **Path Sanitization**: Ensure the UI only reads/writes to approved "Virtual Path" folders (e.g., the current project folder) to prevent the server from accessing sensitive system files.
