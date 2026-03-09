-- =============================================================
-- Brain Tumor Detection System — MySQL Schema
-- =============================================================

CREATE DATABASE IF NOT EXISTS brain_tumor_db;
USE brain_tumor_db;

-- =============================================================
-- ROLE
-- =============================================================
CREATE TABLE role (
    role_id     INT          PRIMARY KEY AUTO_INCREMENT,
    role_name   ENUM('admin','doctor','radiologist','lab_staff','patient') NOT NULL
);

-- =============================================================
-- USER
-- =============================================================
CREATE TABLE user (
    user_id       INT           PRIMARY KEY AUTO_INCREMENT,
    user_name     VARCHAR(100)  NOT NULL,
    email         VARCHAR(100)  NOT NULL UNIQUE,
    password_hash VARCHAR(255)  NOT NULL,
    dob           DATE,
    role_id       INT           NOT NULL,
    created_at    DATETIME      DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_role FOREIGN KEY (role_id) REFERENCES role(role_id)
);

-- =============================================================
-- PHONE NUMBER TYPE
-- =============================================================
CREATE TABLE phone_number_type (
    phone_number_type_id   INT          PRIMARY KEY AUTO_INCREMENT,
    phone_number_type_name VARCHAR(30)  NOT NULL
);

-- =============================================================
-- PHONE NUMBERS
-- =============================================================
CREATE TABLE phone_numbers (
    phone_number_id      INT          PRIMARY KEY AUTO_INCREMENT,
    phone_number         VARCHAR(20)  NOT NULL,
    phone_number_type_id INT          NOT NULL,
    user_id              INT          NOT NULL,
    CONSTRAINT fk_phone_type FOREIGN KEY (phone_number_type_id) REFERENCES phone_number_type(phone_number_type_id),
    CONSTRAINT fk_phone_user FOREIGN KEY (user_id)              REFERENCES user(user_id)
);

-- =============================================================
-- ADDRESS TYPE
-- =============================================================
CREATE TABLE address_type (
    address_type_id   INT         PRIMARY KEY AUTO_INCREMENT,
    address_type_name VARCHAR(30) NOT NULL
);

-- =============================================================
-- ADDRESS
-- =============================================================
CREATE TABLE address (
    address_id      INT          PRIMARY KEY AUTO_INCREMENT,
    user_id         INT          NOT NULL,
    address_type_id INT          NOT NULL,
    street          VARCHAR(150) NOT NULL,
    area            VARCHAR(100),
    city            VARCHAR(100),
    state           VARCHAR(100),
    pincode         VARCHAR(10),
    CONSTRAINT fk_address_user         FOREIGN KEY (user_id)         REFERENCES user(user_id),
    CONSTRAINT fk_address_type         FOREIGN KEY (address_type_id) REFERENCES address_type(address_type_id)
);

-- =============================================================
-- HOSPITAL
-- =============================================================
CREATE TABLE hospital (
    hospital_id   INT          PRIMARY KEY AUTO_INCREMENT,
    hospital_name VARCHAR(150) NOT NULL,
    location      VARCHAR(150),
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================
-- PATIENT
-- =============================================================
CREATE TABLE patient (
    patient_id   INT          PRIMARY KEY AUTO_INCREMENT,
    patient_name VARCHAR(100) NOT NULL,
    patient_dob  DATE         NOT NULL,
    gender       ENUM('male','female','other'),
    user_id      INT          NOT NULL UNIQUE,
    created_at   DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_patient_user FOREIGN KEY (user_id) REFERENCES user(user_id)
);

-- =============================================================
-- DOCTOR
-- =============================================================
CREATE TABLE doctor (
    doctor_id    INT          PRIMARY KEY AUTO_INCREMENT,
    doctor_name  VARCHAR(100) NOT NULL,
    speciality   VARCHAR(100),
    hospital_id  INT          NOT NULL,
    user_id      INT          NOT NULL UNIQUE,
    created_at   DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_doctor_hospital FOREIGN KEY (hospital_id) REFERENCES hospital(hospital_id),
    CONSTRAINT fk_doctor_user     FOREIGN KEY (user_id)     REFERENCES user(user_id)
);

-- =============================================================
-- LAB
-- =============================================================
CREATE TABLE lab (
    lab_id     INT          PRIMARY KEY AUTO_INCREMENT,
    lab_name   VARCHAR(100) NOT NULL,
    user_id    INT          NOT NULL,
    created_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_lab_user FOREIGN KEY (user_id) REFERENCES user(user_id)
);

-- =============================================================
-- RADIOLOGIST
-- =============================================================
CREATE TABLE radiologist (
    radiologist_id   INT          PRIMARY KEY AUTO_INCREMENT,
    radiologist_name VARCHAR(100) NOT NULL,
    user_id          INT          NOT NULL UNIQUE,
    lab_id           INT          NOT NULL,
    created_at       DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_radiologist_user FOREIGN KEY (user_id) REFERENCES user(user_id),
    CONSTRAINT fk_radiologist_lab  FOREIGN KEY (lab_id)  REFERENCES lab(lab_id)
);

-- =============================================================
-- MRI META DATA
-- =============================================================
CREATE TABLE mri_meta_data (
    mri_id       INT          PRIMARY KEY AUTO_INCREMENT,
    patient_id   INT          NOT NULL,
    lab_id       INT          NOT NULL,
    mri_path     VARCHAR(255) NOT NULL,   -- path to H5/NIfTI file
    scan_date    DATE         NOT NULL,
    volume_id    INT,                     -- BraTS volume index
    slice_count  INT,                     -- number of 2D slices
    modality     VARCHAR(10)  DEFAULT 'multi',  -- T1/T2/FLAIR/multi
    notes        TEXT,
    uploaded_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_mri_patient FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
    CONSTRAINT fk_mri_lab     FOREIGN KEY (lab_id)     REFERENCES lab(lab_id)
);

-- =============================================================
-- AI PREDICTIONS
-- =============================================================
CREATE TABLE ai_predictions (
    ai_predictions_id INT          PRIMARY KEY AUTO_INCREMENT,
    mri_id            INT          NOT NULL,
    mask_file_path    VARCHAR(255),        -- predicted segmentation mask image
    heat_map_path     VARCHAR(255),        -- overlay visualization image
    wt_dice           FLOAT,               -- Whole Tumor Dice score
    tc_dice           FLOAT,               -- Tumor Core Dice score
    et_dice           FLOAT,               -- Enhancing Tumor Dice score
    tumor_detected    TINYINT(1)  NOT NULL DEFAULT 0,
    tumor_area_mm2    FLOAT,               -- estimated area in mm²
    estimated_region  VARCHAR(50),         -- e.g. 'Frontal Lobe'
    model_version     VARCHAR(50),         -- checkpoint identifier
    status            ENUM('processing','done','failed') NOT NULL DEFAULT 'processing',
    predicted_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_ai_mri FOREIGN KEY (mri_id) REFERENCES mri_meta_data(mri_id)
);

-- =============================================================
-- RADIOLOGIST REVIEW
-- =============================================================
CREATE TABLE radiologist_review (
    review_id               INT          PRIMARY KEY AUTO_INCREMENT,
    ai_predictions_id       INT          NOT NULL,
    radiologist_id          INT          NOT NULL,
    modified_mask_file_path VARCHAR(255),         -- if radiologist corrects the mask
    diagnosis               VARCHAR(200),         -- final diagnosis text
    review_notes            TEXT,
    status                  ENUM('pending','approved','rejected') NOT NULL DEFAULT 'pending',
    reviewed_at             DATETIME,
    created_at              DATETIME     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_review_ai           FOREIGN KEY (ai_predictions_id) REFERENCES ai_predictions(ai_predictions_id),
    CONSTRAINT fk_review_radiologist  FOREIGN KEY (radiologist_id)    REFERENCES radiologist(radiologist_id)
);

-- =============================================================
-- SEED DATA — roles
-- =============================================================
INSERT INTO role (role_name) VALUES
    ('admin'),
    ('doctor'),
    ('radiologist'),
    ('lab_staff'),
    ('patient');

INSERT INTO phone_number_type (phone_number_type_name) VALUES
    ('mobile'),
    ('home'),
    ('work');

INSERT INTO address_type (address_type_name) VALUES
    ('home'),
    ('work'),
    ('billing');
