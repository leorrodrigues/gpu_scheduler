#ifndef _COMUNICATOR_NOT_INCLUDED_
#define _COMUNICATOR_NOT_INCLUDED_

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdint>

//If we have default rabbit config.
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "../thirdparty/rabbitmq-c/include/amqp_tcp_socket.h"
#include "../thirdparty/rabbitmq-c/include/amqp.h"

#include "common.hpp"
#include "utils.hpp"

//RABBITMQ DEFAULT DEFINES
#define MAX_LISTEN_KEYS 1024
#define LISTEN_KEYS_DELIMITER ","
#define AMQP_CONSUME_MAX_PREFETCH_COUNT 65535

class Comunicator {
private:
struct amqp_connection_info *connInfo;
amqp_connection_state_t conn;
amqp_bytes_t queueName;
amqp_queue_declare_ok_t* res;
int channel;

public:

const char* getMessage(int no_ack = 0) {
	int count=1, prefetch_count=1, local = 0, exclusive = 0;
	/* If there is a limit, set the qos to match */
	//amqp_basic_qos (state, channel, prefetch_size, prefetch_count, global)
	if (count > 0 && count <= AMQP_CONSUME_MAX_PREFETCH_COUNT && !amqp_basic_qos(this->conn, 1, 0, count, 0)) {
		die_rpc(amqp_get_rpc_reply(this->conn), "basic.qos");
	}

	/* if there is a maximum number of messages to be received at a time, set the * qos to match */
	if (prefetch_count > 0 && prefetch_count <= AMQP_CONSUME_MAX_PREFETCH_COUNT) {
		/* the maximum number of messages to be received at a time must be less
		 * than the global maximum number of messages. */
		if (!(count > 0 && count <= AMQP_CONSUME_MAX_PREFETCH_COUNT && prefetch_count >= count)) {
			if (!amqp_basic_qos(conn, 1, 0, prefetch_count, 0)) {
				die_rpc(amqp_get_rpc_reply(conn), "basic.qos");
			}
		}
	}

	amqp_basic_consume_ok_t *result = amqp_basic_consume(this->conn, this->channel, this->queueName, amqp_empty_bytes, local, no_ack, exclusive, amqp_empty_table);
	assert(result != NULL);

	amqp_envelope_t envelope;
	struct timeval timeout = {5, 0};
	amqp_rpc_reply_t rpc_reply = amqp_consume_message(this->conn, &envelope, &timeout, 0);
	assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);

	uint64_t out_body_size_ = envelope.message.body.len;
	char *body = (char *)malloc(sizeof(char)*out_body_size_);
	if (out_body_size_) {
		memcpy(body, envelope.message.body.bytes, out_body_size_);
	}
	//acking message
	if (!no_ack)
		die_amqp_error(amqp_basic_ack(this->conn,  this->channel, envelope.delivery_tag, 0), "basic.ack");

	amqp_destroy_envelope(&envelope);
	return body;
}

/* Convert a amqp_bytes_t to an escaped string form for printing.  We use the same escaping conventions as rabbitmqctl. */
static char *stringifyBytes(amqp_bytes_t bytes) {
	/* We will need up to 4 chars per byte, plus the terminating 0 */
	char *res = (char *) malloc(bytes.len * 4 + 1);
	uint8_t *data = (uint8_t *) bytes.bytes;
	char *p = res;
	size_t i;

	for (i = 0; i < bytes.len; i++) {
		if (data[i] >= 32 && data[i] != 127) {
			*p++ = data[i];
		} else {
			*p++ = '\\';
			*p++ = '0' + (data[i] >> 6);
			*p++ = '0' + (data[i] >> 3 & 0x7);
			*p++ = '0' + (data[i] & 0x7);
		}
	}

	*p = 0;
	return res;
}

public:
Comunicator(const char* user = "guest", const char* password = "guest", const char * host = "localhost", int port=5672, const char* vhost="/", int ssl = 0){
	this->connInfo = (struct amqp_connection_info*) malloc (sizeof(struct amqp_connection_info));
	this->connInfo->user = (char*) user;
	this->connInfo->password = (char*) password;
	this->connInfo->host = (char*) host;
	this->connInfo->port = port;
	this->connInfo->vhost= (char*) vhost;
	this->connInfo->ssl=ssl;
	this->conn = make_connection(this->connInfo);
	this->channel=0;
}

void setup(const char *queue = "test_scheduler", const char *exchange = "amq.direct", const char *routing_key = "task", int declare = 0, int exclusive = 0, int durable = 1, int passive = 0, int autoDelete = 0, int channel = 1) {
	amqp_bytes_t queue_bytes = cstring_bytes(queue);

	char *routing_key_rest;
	char *routing_key_token;
	char *routing_tmp;
	int routing_key_count = 0;
	/* if an exchange name wasn't provided, check that we don't have options that
	 * require it. */
	if (!exchange && !routing_key) {
		fprintf(stderr, "ERROR: --routing-key option requires an exchange name to be provided with --exchange\n");
		exit(1);
	}

	/* Declare the queue as auto-delete.  */
	amqp_queue_declare_ok_t *res = amqp_queue_declare(this->conn, channel, queue_bytes, passive, durable, exclusive, autoDelete, amqp_empty_table);
	assert(res!= NULL);
	if (!res) {
		printf("MERDA\n");
		die_rpc(amqp_get_rpc_reply(conn), "queue.declare");
	}

	if (!queue) {
		/* the server should have provided a queue name */
		char *sq;
		queue_bytes = amqp_bytes_malloc_dup(res->queue);
		sq = stringifyBytes(queue_bytes);
		fprintf(stderr, "Server provided queue name: %s\n", sq);
		free(sq);
	}

	/* Bind to an exchange if requested */
	if (exchange) {
		amqp_bytes_t eb = amqp_cstring_bytes(exchange);

		routing_tmp = strdup(routing_key);
		if (NULL == routing_tmp) {
			fprintf(stderr, "could not allocate memory to parse routing key\n");
			exit(1);
		}

		for (routing_key_token = strtok_r(routing_tmp, LISTEN_KEYS_DELIMITER, &routing_key_rest); NULL != routing_key_token && routing_key_count < MAX_LISTEN_KEYS - 1; routing_key_token = strtok_r(NULL, LISTEN_KEYS_DELIMITER, &routing_key_rest)) {
			if (!amqp_queue_bind(conn, 1, queue_bytes, eb, cstring_bytes(routing_key_token), amqp_empty_table)) {
				die_rpc(amqp_get_rpc_reply(conn), "queue.bind");
			}
		}
		free(routing_tmp);

	}
	this->channel=channel;
	this->queueName=queue_bytes;
	this->res=res;
}

void closeConnection() {
	close_connection(this->conn);
}

inline const char* getNextTask(){
	return getMessage();
}

inline void getNTasks(int n){
	for(int i=0; i<n; i++) {
		std::cout<<"Getting message "<<i+1<<"\n";
		const char* message= getNextTask();
		//check if message comes empty by some erro in getMessage.
		if(message!="") std::cout<<message<<"\n------\n";
		else i--;
	}
}

inline int getQueueSize(){
	return this->res->message_count;
}

};
#endif
