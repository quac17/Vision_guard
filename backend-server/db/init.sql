--
-- PostgreSQL database dump
--

\restrict yZA1j5UXRBFeh55n3EJDDVfUza2Kw4r0HLU79RSiCfEWVkwFdl1xdTEsjT2x1SP

-- Dumped from database version 15.16
-- Dumped by pg_dump version 15.16

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: attendance; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.attendance (
    id integer NOT NULL,
    student_id integer,
    status character varying(20) NOT NULL,
    attendance_time timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    notified integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT attendance_status_check CHECK (((status)::text = ANY ((ARRAY['on_bus'::character varying, 'off_bus'::character varying])::text[])))
);


ALTER TABLE public.attendance OWNER TO postgres;

--
-- Name: attendance_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.attendance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.attendance_id_seq OWNER TO postgres;

--
-- Name: attendance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.attendance_id_seq OWNED BY public.attendance.id;


--
-- Name: student_photos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.student_photos (
    id integer NOT NULL,
    student_id integer,
    photo_path character varying(255) NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.student_photos OWNER TO postgres;

--
-- Name: student_photos_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.student_photos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.student_photos_id_seq OWNER TO postgres;

--
-- Name: student_photos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.student_photos_id_seq OWNED BY public.student_photos.id;


--
-- Name: students; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.students (
    id integer NOT NULL,
    full_name character varying(100) NOT NULL,
    grade character varying(20) NOT NULL,
    student_code character varying(50) NOT NULL,
    parent_id integer,
    email character varying(100),
    phone character varying(20),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.students OWNER TO postgres;

--
-- Name: students_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.students_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.students_id_seq OWNER TO postgres;

--
-- Name: students_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.students_id_seq OWNED BY public.students.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    hashed_password character varying(255) NOT NULL,
    role character varying(20) NOT NULL,
    email character varying(100),
    phone character varying(20),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT users_role_check CHECK (((role)::text = ANY ((ARRAY['admin'::character varying, 'parent'::character varying])::text[])))
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.users_id_seq OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: attendance id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendance ALTER COLUMN id SET DEFAULT nextval('public.attendance_id_seq'::regclass);


--
-- Name: student_photos id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student_photos ALTER COLUMN id SET DEFAULT nextval('public.student_photos_id_seq'::regclass);


--
-- Name: students id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.students ALTER COLUMN id SET DEFAULT nextval('public.students_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Data for Name: attendance; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.attendance (id, student_id, status, attendance_time, notified, created_at) FROM stdin;
1	1	on_bus	2026-02-22 17:29:16.806974+00	0	2026-02-22 19:29:16.806974+00
2	1	off_bus	2026-02-22 18:29:16.809204+00	0	2026-02-22 19:29:16.809204+00
\.


--
-- Data for Name: student_photos; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.student_photos (id, student_id, photo_path, created_at) FROM stdin;
\.


--
-- Data for Name: students; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.students (id, full_name, grade, student_code, parent_id, email, phone, created_at) FROM stdin;
1	Nguyễn Văn A	1A	SV001	2	viethoangtran171@gmail.com	0123456789	2026-02-22 19:29:16.803234+00
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, username, hashed_password, role, email, phone, created_at) FROM stdin;
1	admin	$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6L6.6p6p6p6p6p6	admin	admin@visionguard.com	\N	2026-02-22 19:29:16.797748+00
2	parent_test	$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6L6.6p6p6p6p6p6	parent	parent@example.com	0123456789	2026-02-22 19:29:16.801251+00
\.


--
-- Name: attendance_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.attendance_id_seq', 2, true);


--
-- Name: student_photos_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.student_photos_id_seq', 1, false);


--
-- Name: students_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.students_id_seq', 1, true);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 2, true);


--
-- Name: attendance attendance_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendance
    ADD CONSTRAINT attendance_pkey PRIMARY KEY (id);


--
-- Name: student_photos student_photos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student_photos
    ADD CONSTRAINT student_photos_pkey PRIMARY KEY (id);


--
-- Name: students students_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_pkey PRIMARY KEY (id);


--
-- Name: students students_student_code_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_student_code_key UNIQUE (student_code);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: attendance attendance_student_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendance
    ADD CONSTRAINT attendance_student_id_fkey FOREIGN KEY (student_id) REFERENCES public.students(id) ON DELETE CASCADE;


--
-- Name: student_photos student_photos_student_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student_photos
    ADD CONSTRAINT student_photos_student_id_fkey FOREIGN KEY (student_id) REFERENCES public.students(id) ON DELETE CASCADE;


--
-- Name: students students_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict yZA1j5UXRBFeh55n3EJDDVfUza2Kw4r0HLU79RSiCfEWVkwFdl1xdTEsjT2x1SP

