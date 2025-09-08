pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/PHAMXUANKHIEM/project1'
            }
        }

        stage('Setup Python Env') {
            steps {
                sh '''
                    python3 -m venv venv
                    source venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt || true
                '''
            }
        }

        stage('Run Python Script') {
            steps {
                sh '''
                    source venv/bin/activate
                    python app.py
                '''
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished!'
        }
    }
}
