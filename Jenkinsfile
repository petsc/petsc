#!/usr/bin/env groovy
def alljob = JOB_NAME.tokenize('/') as String[]
def node_name = alljob[0]
def arch_name = alljob[1]

pipeline { 
    agent { node { label node_name} }
    
    stages {
        stage('Local Merge') {
            steps {
                checkout scm
            }
        }
        stage('Configure') {
            steps {
                sh "python ./config/examples/${arch_name}.py"
            }
        }
        stage('Make') {
            steps {
                sh "make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} all"
            }
        }
        stage('Examples') {
            steps {
                sh "make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} -f gmakefile test"    
            }
        }
    }  
}
