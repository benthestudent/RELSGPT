import ChatBot from 'react-chatbotify';
import React from "react";

const ASK_ENDPOINT = "http://127.0.0.1:5000/ask?"

const Chat = () => {
    async function askQuestion(q) {
        try {
            const response = await fetch(ASK_ENDPOINT + new URLSearchParams({
                question: q.userInput
            }));
            const data = await response.text();
            await q.injectMessage(data);
        } catch (error) {
            console.log(error);
            return "Something went wrong";
        }
    };
    const flow={
        start: {
            message: "Do you have any questions about religious studies?",
            path: "loop"
        },
        loop: {
            message: async (params) => {
                const result = await askQuestion(params);
                return result;
            },
            path: "loop",
        }
    };

    return (
        <ChatBot options={
            {theme: {
                embedded: true,
                primaryColor: "#35A9FF",
                secondaryColor: "#35A9FF"
            }, 
            botBubble: {dangerouslySetInnerHtml: true},
            header: {title: "RELSGPT"},
            width: '500px'
            }} flow={flow}
            />
    );
}


export default Chat;