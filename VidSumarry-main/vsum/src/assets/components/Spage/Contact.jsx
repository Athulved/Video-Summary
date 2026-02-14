import { useForm } from "react-hook-form";
import emailjs from "@emailjs/browser";
import "./Contact.css";
import Navbar from "./Navbar";

export default function ContactUs() {
  const { register, handleSubmit, formState: { errors } } = useForm();

  const onSubmit = (data) => {
    emailjs.send(
      "service_4zrt9ot",  // Replace with your EmailJS Service ID
      "template_tmm11xa", // Replace with your EmailJS Template ID
      {
        from_name: data.name, // Sender's Name
        email: data.email,
        message: data.message,
      },
      "7h_DSlMsG7NWJLaPX"   // Replace with your EmailJS Public Key
    ).then(() => {
      alert("Your message has been sent successfully!");
    }).catch((error) => {
      console.error("Email sending error:", error);
      alert("Failed to send message. Please try again.");
    });
  };

  return (
    <div className="container">
      <Navbar/>
      <div className="contact-wrapper">
        <div className="card">
          <h2 className="title">Contact Us</h2>
          <form onSubmit={handleSubmit(onSubmit)} className="form">
            <div className="form-group">
              <label>Name</label>
              <input type="text" placeholder="Your Name" {...register("name", { required: "Name is required" })} />
              {errors.name && <p className="error">{errors.name.message}</p>}
            </div>

            <div className="form-group">
              <label>Email</label>
              <input
                type="email"
                placeholder="Your Email"
                {...register("email", {
                  required: "Email is required",
                  pattern: {
                    value: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/,
                    message: "Invalid email address",
                  },
                })}
              />
              {errors.email && <p className="error">{errors.email.message}</p>}
            </div>

            <div className="form-group">
              <label>Message</label>
              <textarea placeholder="Your Message" {...register("message", { required: "Message is required" })}></textarea>
              {errors.message && <p className="error">{errors.message.message}</p>}
            </div>

            <button type="submit" className="submit-btn">Send Message</button>
          </form>
        </div>

        {/* Google Map Section */}
        <div className="map-container">
          <iframe
            src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3926.372418059777!2d76.40626477535238!3d10.23151758988602!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3b08068aa17bd247%3A0xf048b9ebcbd2af28!2sFederal%20Institute%20of%20Science%20And%20Technology%20(FISAT)%C2%AE!5e0!3m2!1sen!2sin!4v1741287652305!5m2!1sen!2sin"
            width="100%"
            height="100%"
            style={{ border: 0 }}
            allowFullScreen=""
            loading="lazy"
          ></iframe>
          
        </div>
      </div>
    </div>
  );
}
