export const getAIMessage = async (userQuery) => {
  try {
    const response = await fetch('http://localhost:5000/api/chat', 
      { method: "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query : userQuery, new_thread : true })
      });

    const data = await response.json()

    const message = {
      role: "assistant",
      content: data.answer
    }

    return message
  }
  catch (error) {
    console.error("failed to call Flask server");
    const message = {
      role: "assistant",
      content: "error in calling Flask server"
    };

    return message
  }
};
