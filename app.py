from flask import Flask, render_template, request, url_for
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt


import joblib  # for loading a pre-trained model

app = Flask(__name__)

k_fit_50 = joblib.load('big_five_traits.pkl')
scaler = joblib.load('scaler.pkl') 


@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/personnality')
def personnality():
    return render_template('personnality.html')

@app.route('/ml', methods=['GET'])
def ml():
    return render_template('ml.html')

@app.route('/question')
def question():
    return render_template('question.html')

@app.route('/email')
def email():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    
    ext1 = int(request.form.get('EXT1'))
    ext2 = int(request.form.get('EXT2'))
    ext3 = int(request.form.get('EXT3'))
    ext4 = int(request.form.get('EXT4'))
    ext5 = int(request.form.get('EXT5'))
    ext6 = int(request.form.get('EXT6'))
    ext7 = int(request.form.get('EXT7'))
    ext8 = int(request.form.get('EXT8'))
    ext9 = int(request.form.get('EXT9'))
    ext10 = int(request.form.get('EXT10'))
    ext_score = np.mean([ext1, ext2, ext3, ext4, ext5, ext6, ext7, ext8, ext9, ext10])
    ext_percentage  = round((ext_score / 5) * 100, 2)

    agr1 = int(request.form.get('AGR1'))
    agr2 = int(request.form.get('AGR2'))
    agr3 = int(request.form.get('AGR3'))
    agr4 = int(request.form.get('AGR4'))
    agr5 = int(request.form.get('AGR5'))
    agr6 = int(request.form.get('AGR6'))
    agr7 = int(request.form.get('AGR7'))
    agr8 = int(request.form.get('AGR8'))
    agr9 = int(request.form.get('AGR9'))
    agr10 = int(request.form.get('AGR10'))
    agr_score = np.mean([agr1, agr2, agr3, agr4, agr5, agr6, agr7, agr8, agr9, agr10])
    agr_percentage  = round((agr_score / 5) * 100, 2)

    csn1 = int(request.form.get('CSN1'))
    csn2 = int(request.form.get('CSN2'))
    csn3 = int(request.form.get('CSN3'))
    csn4 = int(request.form.get('CSN4'))
    csn5 = int(request.form.get('CSN5'))
    csn6 = int(request.form.get('CSN6'))
    csn7 = int(request.form.get('CSN7'))
    csn8 = int(request.form.get('CSN8'))
    csn9 = int(request.form.get('CSN9'))
    csn10 = int(request.form.get('CSN10'))
    csn_score = np.mean([csn1, csn2, csn3, csn4, csn5, csn6, csn7, csn8, csn9, csn10])
    csn_percentage  = round((csn_score / 5) * 100, 2)

    est1 = int(request.form.get('EST1'))
    est2 = int(request.form.get('EST2'))
    est3 = int(request.form.get('EST3'))
    est4 = int(request.form.get('EST4'))
    est5 = int(request.form.get('EST5'))
    est6 = int(request.form.get('EST6'))
    est7 = int(request.form.get('EST7'))
    est8 = int(request.form.get('EST8'))
    est9 = int(request.form.get('EST9'))
    est10 = int(request.form.get('EST10'))
    est_score = np.mean([est1, est2, est3, est4, est5, est6, est7, est8, est9, est10])
    est_percentage  = round((est_score / 5) * 100, 2)

    opn1 = int(request.form.get('OPN1'))
    opn2 = int(request.form.get('OPN2'))
    opn3 = int(request.form.get('OPN3'))
    opn4 = int(request.form.get('OPN4'))
    opn5 = int(request.form.get('OPN5'))
    opn6 = int(request.form.get('OPN6'))
    opn7 = int(request.form.get('OPN7'))
    opn8 = int(request.form.get('OPN8'))
    opn9 = int(request.form.get('OPN9'))
    opn10 = int(request.form.get('OPN10'))
    opn_score = np.mean([opn1, opn2, opn3, opn4, opn5, opn6, opn7, opn8, opn9, opn10])
    opn_percentage  = round((opn_score / 5) * 100, 2)

    print(f"opn_score: {opn_score}")

    Traits = np.array([
    ext1, ext2, ext3, ext4, ext5, ext6, ext7, ext8, ext9, ext10,
    agr1, agr2, agr3, agr4, agr5, agr6, agr7, agr8, agr9, agr10,
    csn1, csn2, csn3, csn4, csn5, csn6, csn7, csn8, csn9, csn10,
    est1, est2, est3, est4, est5, est6, est7, est8, est9, est10,
    opn1, opn2, opn3, opn4, opn5, opn6, opn7, opn8, opn9, opn10])

    Traits = Traits.reshape(1,-1)
    user_responses_scaled = scaler.transform(Traits)

    Average_traits = np.array([ext_score, agr_score, csn_score, est_score, opn_score])

    
    
  

    

    
    
    print(f"user_responses_scaled : {user_responses_scaled }")

     # Predict the user's cluster
    user_cluster = k_fit_50.predict(user_responses_scaled)[0]
    cluster = user_cluster
    print(f"Scaled Responses: {user_responses_scaled}")
    print(f"Predicted Cluster: {user_cluster}")
    
    # Create histogram for Big Five traits
        
    fig, ax = plt.subplots()
    ax.bar(['EXT', 'AGR', 'CSN', 'EST', 'OPN'], 
               [ext_score, agr_score, csn_score, est_score, opn_score], color='skyblue')
    ax.set_xlabel('Personality Traits')
    ax.set_ylabel('Scores')
    ax.set_title('Big Five Personality Scores')
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

   

    cluster_advice = {
        1: {
            "title": "Cluster 1 : The Gentle Guardian",
            "description": "Your spirit animal is an elephant. Elephants are highly social, intelligent, and known for their empathy. Their strong sense of responsibility for their herd and their calm, measured approach to life make them a fitting representation of this cluster. While they appreciate social connections, they also have a grounded sense of routine.",
        },
        2: {
            "title": "Cluster 2 : The Solitary Wanderer",
            "description": "Your spirit animal is a cheetah. Cheetahs are solitary hunters, preferring their own space. They are calm and emotionally stable but less focused on social interaction. Their need for independence and freedom from constraints, paired with a lack of interest in novelty or change, makes them a good match for this cluster.",
        },
        3: {
            "title": "Cluster 3 : The Balanced Companion",
            "description": "Your spirit animal is a dolphin. Dolphins are known for their playful, social nature and their ability to form strong bonds with others. They are intelligent and adaptable, balancing their openness to new experiences with a sense of responsibility within their pods. While they are generally calm, they can be sensitive in stressful situations.",
        },
        4: {
            "title": "Cluster 4 : The Energetic Innovator",
            "description": "Your spirit animal is a parrot. Parrots are highly social and talkative, often thriving in dynamic environments. They are open to new ideas and experiences, bringing an energetic and innovative spirit to the table. However, they can be sensitive and prone to emotional reactions, reflecting the higher neuroticism of this cluster.",
        }, 
        5: {
            "title": "Cluster 5 : The Bold Adventurer",
            "description": "Your spirit animal is a tiger. Tigers embody boldness and independence. They are fearless explorers, often leading a solitary life, but their emotional intensity and occasional volatility capture the high neuroticism of this cluster. They thrive in dynamic, challenging environments and aren't afraid of taking risks.",
        }, 
        6: {
            "title": "Cluster 6 : The Steady Collaborator",
            "description": "Your spirit animal is a dog. Dogs are known for their loyalty, responsibility, and deep sense of companionship. They balance social engagement with calmness and emotional resilience. They also demonstrate dependability and are quick to cooperate with others, reflecting the conscientiousness and agreeableness in this cluster.",
        }, 
        7: {
            "title": "Cluster 7 : The reflective connector",
            "description": "Your spirit animal is an owl. Owls are reflective, wise, and observant. They represent a balance of openness to new ideas and a slightly reserved but thoughtful approach to life. While they are social in certain environments, they also have a tendency to be introspective and sensitive, particularly when faced with stress or uncertainty.",
        }, 
    }

    # Map clusters to image filenames
    cluster_images = {
        1: url_for('static', filename='images/elephant.jpg'),
        2: url_for('static', filename='images/cheetah.jpg'),
        3: url_for('static', filename='images/dolphin.jpg'),
        4: url_for('static', filename='images/parrot.jpg'),
        5: url_for('static', filename='images/tiger.jpg'),
        6: url_for('static', filename='images/dog.jpg'),
        7: url_for('static', filename='images/owl.jpg'),
    }

    # Get the image filename for the cluster
    cluster_image = cluster_images.get(cluster, None)
    cluster_image_url = cluster_images[user_cluster]

    
    

    if ext_percentage < 50:
        ext_advice = (
        "It seems like you're more introverted, which means you may find energy in quieter, reflective moments. "
        "Taking time for yourself to recharge is essential for your well-being, but don’t hesitate to step outside your comfort zone occasionally. "
        "Engaging in social activities that align with your interests, even in small groups, can provide a healthy balance and new opportunities for growth."
    )
    elif 50 <= ext_percentage <= 75:
        ext_advice = (
        "You have a balanced level of extraversion, which allows you to enjoy the best of both worlds. "
        "You likely appreciate social interactions when they feel meaningful or fun but also value your alone time to recharge. "
        "This balance can be a strength, so make time for both engaging with others and nurturing your inner self to maintain harmony."
    )
    else:
        ext_advice = (
        "You are highly extroverted and thrive in social environments, gaining energy from interactions with others. "
        "Continue embracing your social nature by seeking opportunities to connect, collaborate, and share your ideas with those around you. "
        "Just remember to occasionally take a step back to reflect and recharge, ensuring your energy remains sustainable and fulfilling."
    )


    if agr_percentage < 50:
        agr_advice = (
        "You may lean towards being more competitive or assertive, which can be advantageous in pursuing personal goals. "
        "However, consider how empathy and cooperation could enhance your relationships and teamwork. "
        "Practicing active listening and seeking win-win solutions in conflicts can help strengthen your social connections."
    )
    elif 50 <= agr_percentage <= 75:
        agr_advice = (
        "You have a balanced approach to agreeableness, showing empathy and understanding while also standing up for your beliefs. "
        "This allows you to navigate social dynamics effectively and maintain strong relationships. "
        "Continue fostering this balance by considering others’ perspectives while remaining true to your values."
    )
    else:
        agr_advice = (
        "You are highly agreeable, which makes you warm, cooperative, and easy to get along with. "
        "Your focus on harmony in relationships is admirable, but be mindful not to prioritize others' needs at the expense of your own. "
        "Setting healthy boundaries can ensure that your kindness remains sustainable and reciprocated."
    )

    
    if csn_percentage < 50:
        csn_advice = (
        "You might find it challenging to stay organized or focused, but this is an area you can improve. "
        "Start by setting small, achievable goals and breaking larger tasks into manageable steps. "
        "Creating routines or using tools like to-do lists and reminders can help enhance your ability to follow through effectively."
    )
    elif 50 <= csn_percentage <= 75:
        csn_advice = (
        "You are generally responsible and goal-oriented, with a good sense of how to stay on track. "
        "While you manage tasks effectively, there may still be room to refine your attention to detail or improve consistency in some areas. "
        "Building on your strengths by fine-tuning time management or prioritization can help you reach even higher levels of efficiency."
    )
    else:
        csn_advice = (
        "You are highly conscientious, showing excellent organization, reliability, and attention to detail. "
        "Your strong sense of responsibility makes you well-equipped to handle complex tasks and long-term goals. "
        "While your conscientiousness is a strength, be cautious about overloading yourself or striving for perfection, as this can lead to burnout.\n"

        "Interesting fact : Conscientiousness is the strongest predictor of all five traits for job performance (John & Srivastava, 1999)."
    )

    
    if est_percentage < 50:
        est_advice = (
        "You are emotionally stable, which allows you to manage stress effectively and maintain a calm demeanor. "
        "This resilience is a valuable asset in both personal and professional settings. "
        "Continue embracing positive coping strategies and a balanced mindset to maintain your emotional well-being."
    )
    elif 50 <= est_percentage <= 75:
        est_advice = (
        "You experience some emotional fluctuations but generally handle them well. "
        "Strengthening emotional regulation skills, such as mindfulness or stress management techniques, can enhance your overall stability. "
        "This balanced emotional awareness can be a strength in understanding both your own and others' feelings."
    )
    else:
        est_advice = (
        "You may experience heightened emotional stress or sensitivity, which can feel overwhelming at times. "
        "Consider engaging in activities that promote relaxation, such as meditation, yoga, or regular physical exercise. "
        "If emotions feel too intense to manage alone, seeking support from a trusted friend, mentor, or professional could make a significant difference."
    )

    
    if opn_percentage < 50:
        opn_advice = (
        "You may prefer familiar routines and experiences, finding comfort in the tried and true. "
        "While this preference for stability has its benefits, it can be rewarding to occasionally explore new perspectives or try creative activities. "
        "Stepping outside your comfort zone, even in small ways, can lead to personal growth and unexpected opportunities."
    )
    elif 50 <= opn_percentage <= 75:
        opn_advice = (
        "You are open to new experiences and enjoy exploring creative or innovative ideas. "
        "This flexibility allows you to adapt well to change while still staying grounded. "
        "Balancing openness with practicality can help you turn your curiosity and creativity into meaningful outcomes."
    )
    else:
        opn_advice = (
        "You are highly open to new ideas and experiences, thriving on creativity, exploration, and intellectual pursuits. "
        "This curiosity can lead to profound insights and novel opportunities, but ensure you stay connected to what is meaningful and practical for your goals. "
        "Focusing your openness toward areas of personal significance can help you make the most of this trait."
    )

# Get advice for the specific cluster or provide a default message
    advice = cluster_advice.get(cluster, {"title": "Unknown Cluster", "description": "No specific advice available."})


    

        # Pass the cluster and chart to the template
      
    return render_template('submit.html',
            ext_score=ext_score,
            ext_percentage=ext_percentage,
            agr_score=agr_score,
            agr_percentage=agr_percentage,
            csn_score=csn_score,
            csn_percentage=csn_percentage,
            est_score=est_score,
            est_percentage=est_percentage,
            opn_score=opn_score,
            opn_percentage=opn_percentage,
            cluster=user_cluster, 
            img_base64 = img_base64,
            
            title=advice["title"], description=advice["description"],
            ext_advice= ext_advice,
            agr_advice=agr_advice,
            csn_advice=csn_advice,
            est_advice=est_advice,
            opn_advice=opn_advice,
             cluster_image= cluster_image,
             cluster_image_url =cluster_image_url )


      
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
